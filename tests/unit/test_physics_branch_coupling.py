"""Finite real component output/consumer checks, without a SECHIBA model step.

The energy component has independent same-input Fortran truth. HYDROL is only
its real, deliberately incomplete pre-call assembler here. The canopy chain
checks actual input selection and writeback; it adds no energy-oracle claim.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from jax_orchidee.driver import orchestration as driver
from jax_orchidee.sechiba.coupling import enerbil_downstream_payload
from jax_orchidee.sechiba.enerbil import EnerbilFirstStepLocalAssembly
from jax_orchidee.sechiba.hydrol import assemble_hydrol_first_step_precall_payload

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _hydrol_boundary(payload, veget_max):
    """Keep unrelated HYDROL prerequisites absent, rather than invent state."""
    return assemble_hydrol_first_step_precall_payload(
        enerbil_payload=payload,
        hydrol_restart=SimpleNamespace(),
        slowproc_restart=SimpleNamespace(),
        pref_soil_veg=np.ones(14, dtype=np.int32),
        nstm=1,
        ext_coeff_vegetfrac=np.full(14, 0.5),
        static_payload_template={"veget_max": np.asarray(veget_max)},
    )


@pytest.mark.parametrize("mode", ("eager", "jit"))
@pytest.mark.parametrize("case_index", (0, 1))
def test_energy_original_outputs_reach_next_boundary_and_unit_writer(case_index, mode):
    from scripts.dev.oracle_physics_branch_energy import run_production

    reference = json.loads((FIXTURES / "physics_branch_energy_reference.json").read_text(encoding="utf-8"))
    rate_reference = json.loads((FIXTURES / "physics_branch_coupling_reference.json").read_text(encoding="utf-8"))
    case = reference["cases"][case_index]
    inputs, expected = case["inputs"], case["expected"]
    result = run_production(inputs, mode)
    downstream = enerbil_downstream_payload(result)
    tolerance = {k: reference["tolerance"][k] for k in ("rtol", "atol")}
    for name, value in downstream.items():
        actual = np.asarray(value)
        if actual.ndim == 2:
            actual = actual[:, 13]
        np.testing.assert_allclose(actual, expected[name], **tolerance, err_msg=name)

    hydro = _hydrol_boundary(downstream, inputs["veget_max"])
    assert hydro.enerbil_boundary.ok
    assert hydro.missing_inputs  # This is not a ready/full HYDROL execution.
    for name in hydro.enerbil_inputs:
        np.testing.assert_array_equal(hydro.payload[name], downstream[name])

    # The typed shell packages an actual component result. It does not claim
    # that run_enerbil_first_step_local_from_precall made this different tuple.
    # JSON is a transport format; the actual driver carries numerical arrays.
    payload = {
        **{name: np.asarray(value) if isinstance(value, list) else value for name, value in inputs.items()},
        **downstream,
        "zlev": np.full(8, 2.0),
        "vevapp": result.flux.vevapp,
        "fluxsens": result.flux.fluxsens,
    }
    local = EnerbilFirstStepLocalAssembly(
        step=result, payload=payload, source_kernel_inputs=tuple(downstream),
        passthrough_inputs=tuple(inputs),
        provenance=("Actual original-tuple local ENERBIL result; packaging only",),
    )
    fields = driver._empty_previous_step_state_fields()
    driver._add_enerbil_local_to_previous_fields(fields, local)
    rate = rate_reference["cases"][case_index]
    assert rate["name"] == case["name"]
    assert rate["dt_sechiba"] == inputs["dt_sechiba"]
    np.testing.assert_allclose(payload["vevapp"], rate["amount"], **tolerance)
    np.testing.assert_allclose(fields["driver_previous_step_state"]["vevapp"], rate["rate"], **tolerance)
    assert np.any(np.abs(np.asarray(rate["amount"]) - np.asarray(rate["rate"])) > 1e-4)
    mapped, _, missing = driver._diffuco_restart_payload_from_state_fields(
        diffuco_state=fields["diffuco_previous_step_state"], hydrol_state={}, slowproc_state={},
    )
    assert missing  # A selected-field mapper check is not a complete next step.
    for target, source in (("temp_sol", "temp_sol_new"), ("temp_sol_pft", "temp_sol_new_pft"),
                           ("qsurf", "qsurf"), ("evapot", "evapot"), ("evapot_corr", "evapot_corr")):
        np.testing.assert_array_equal(mapped[target], payload[source])


@pytest.mark.parametrize("use_jit", (False, True))
def test_new_above_floor_canopy_case_reaches_actual_energy_and_next_hydrol(use_jit):
    from scripts.dev.oracle_canopy_component_fixture import run_canopy_components
    from scripts.dev.oracle_physics_branch_canopy import make_canopy_fixture

    fixture = make_canopy_fixture()
    index = fixture.case_names.index("water100")

    def selected(payload):
        return {name: np.asarray(value)[index:index + 1] for name, value in payload.items()}

    fixture = fixture._replace(
        case_names=("water100",),
        diffuco_precall=fixture.diffuco_precall._replace(payload=selected(fixture.diffuco_precall.payload)),
        driver=selected(fixture.driver), albedo=selected(fixture.albedo), thermal=selected(fixture.thermal),
    )
    assert float(fixture.diffuco_precall.payload["humrel"][0, 13]) == 1.0
    components = run_canopy_components(fixture, diffuco_use_jit=use_jit, enerbil_use_jit=use_jit)
    assert not jax.config.jax_disable_jit
    assert components.enerbil_precall.ok
    after_diffuco = components.diffuco.result.payload.payload
    for name in components.enerbil_precall.diffuco_inputs:
        np.testing.assert_array_equal(components.enerbil_precall.payload[name], after_diffuco[name])
    np.testing.assert_array_equal(components.enerbil.payload["vbeta3"], after_diffuco["vbeta3"])
    np.testing.assert_array_equal(components.enerbil.payload["q_cdrag"], after_diffuco["q_cdrag"])
    np.testing.assert_array_equal(components.enerbil.payload["transpir"], components.enerbil.step.evapveg_pft.transpir)
    assert float(np.asarray(components.enerbil.payload["transpir"])[0, 13]) > 0
    hydro = _hydrol_boundary(components.enerbil.payload, after_diffuco["veget_max"])
    assert hydro.enerbil_boundary.ok and hydro.missing_inputs
    for name in hydro.enerbil_inputs:
        np.testing.assert_array_equal(hydro.payload[name], components.enerbil.payload[name])
