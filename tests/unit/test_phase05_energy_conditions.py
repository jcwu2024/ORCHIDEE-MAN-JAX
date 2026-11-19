"""Current production energy conditions with original-source frozen outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.dev import oracle_phase05_energy_conditions as oracle

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def reference():
    return json.loads(oracle.FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def checks(reference):
    return oracle.certify(reference)[0]


@pytest.mark.parametrize("mode", ("eager", "jit"))
@pytest.mark.parametrize("case", ("joint", "snow_conjuncts", "snow_disabled", "tiny_milly_local"))
def test_current_production_against_original_fortran(checks, case, mode):
    selected = [row for row in checks if row["case"] == case and row["mode"] == mode]
    assert selected
    assert all(row["passed"] for row in selected), [row for row in selected if not row["passed"]]


def test_complete_declared_inputs_bound_to_frozen_reference(reference):
    assert oracle.joint_inputs(oracle.joint_fixture()) == reference["joint"]["inputs"]
    for case, tiny, explicit in zip(reference["flux"], (False, False, True), (True, False, True), strict=True):
        assert oracle.jsonable(oracle.make_flux_inputs(tiny=tiny, explicit=explicit)) == case["inputs"]


def test_original_guard_observations_have_required_sides(reference):
    cases = {case["name"]: case for case in reference["flux"]}
    snow = cases["snow_conjuncts"]
    assert snow["inputs"]["temp_sol_new"][:3] == [274.0, 273.15, 272.0]
    assert snow["inputs"]["soilcap"][4:6] == [0.0, -1.0]
    assert snow["expected"]["temp_sol_add"][0] != 0.0
    assert snow["expected"]["temp_sol_add"][1:6] == [0.0] * 5
    disabled = cases["snow_disabled"]
    assert disabled["expected"]["pgflux"] == disabled["inputs"]["pgflux"]
    tiny = cases["tiny_milly_local"]["expected"]
    assert tiny["warning"] == [1.0, 0.0, 0.0, 0.0]
    assert 0.0 < tiny["correction"][0] <= 1e-8
    # The neighbor takes the divide branch; the returned value is the final
    # ratio, not the pre-division denominator tested against min_sechiba.
    assert 0.0 < tiny["correction"][1] < tiny["correction"][0]
    assert tiny["evapot"][2] == 0.0
    assert tiny["correction"][2:] == [0.0, 0.0]


@pytest.mark.parametrize("use_jit", (False, True))
def test_actual_joint_predecessor_return_writer_and_next_boundary(reference, use_jit):
    from jax_orchidee.driver import orchestration as driver
    from jax_orchidee.sechiba.coupling import enerbil_downstream_payload
    from jax_orchidee.sechiba.hydrol import assemble_hydrol_first_step_precall_payload

    result = oracle.run_joint(reference["joint"]["inputs"], use_jit=use_jit)
    assert result.enerbil_precall.ok
    diff = result.diffuco.result.payload.payload
    # These values are actual joint DIFFUCO outputs, not independently selected betas.
    np.testing.assert_array_equal(diff["valpha"], np.ones(4))
    np.testing.assert_allclose(
        diff["vbeta"],
        np.asarray(diff["vbeta4"]) + np.sum(diff["vbeta2"], axis=1) + np.sum(diff["vbeta3"], axis=1),
        **oracle.TOLERANCE,
    )
    for name in ("vbeta", "vbeta1", "vbeta2", "vbeta3", "vbeta4", "vbeta5"):
        assert np.all(np.asarray(diff[name]) >= 0.0)
    for name in ("vbeta1", "vbeta5"):
        assert np.all(np.asarray(diff[name]) <= 1.0)
    for name in (
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
        "q_cdrag",
        "q_cdrag_pft",
    ):
        np.testing.assert_array_equal(result.enerbil.payload[name], diff[name])
    speed = np.maximum(0.1, np.hypot(result.fixture.driver["u"], result.fixture.driver["v"]))
    assert np.all(speed * np.asarray(diff["q_cdrag"]) >= 1e-4)
    downstream = enerbil_downstream_payload(result.enerbil.step)
    for name, actual in downstream.items():
        expected = reference["joint"]["expected"]["enerbil." + name]
        actual = np.asarray(actual)
        if name == "temp_sol_new_pft":
            actual = actual[:, 1:]
        np.testing.assert_allclose(actual.ravel(), expected, **oracle.TOLERANCE, err_msg=name)
    # Real production writer called on its actual wrapper output, without a
    # synthesized shell. The rate boundary's independent original amount is
    # retained; prior original intersurf /dt evidence supplies that conversion.
    fields = driver._empty_previous_step_state_fields()
    driver._add_enerbil_local_to_previous_fields(fields, result.enerbil)
    amount = reference["joint"]["expected"]["enerbil.vevapp"]
    dt = result.fixture.static_kwargs["dt_sechiba"]
    np.testing.assert_allclose(
        fields["driver_previous_step_state"]["vevapp"], np.asarray(amount) / dt, **oracle.TOLERANCE
    )
    for target, source in (
        ("temp_sol", "temp_sol_new"),
        ("temp_sol_pft", "temp_sol_new_pft"),
        ("qsurf", "qsurf"),
        ("evapot", "evapot"),
        ("evapot_corr", "evapot_corr"),
    ):
        np.testing.assert_array_equal(fields["diffuco_previous_step_state"][target], result.enerbil.payload[source])
    hydro = assemble_hydrol_first_step_precall_payload(
        enerbil_payload=downstream,
        hydrol_restart=SimpleNamespace(),
        slowproc_restart=SimpleNamespace(),
        pref_soil_veg=np.ones(14, dtype=np.int32),
        nstm=1,
        ext_coeff_vegetfrac=np.full(14, 0.5),
        static_payload_template={"veget_max": np.asarray(diff["veget_max"])},
    )
    assert hydro.enerbil_boundary.ok and hydro.missing_inputs
    for name in hydro.enerbil_inputs:
        np.testing.assert_array_equal(hydro.payload[name], downstream[name])


@pytest.mark.external_data
def test_original_reference_source_and_verbatim_extraction(reference):
    missing = sorted({s["path"] for s in reference["source"] if not (ROOT / s["path"]).is_file()})
    if missing:
        pytest.skip("Original source/span gate NOT EXECUTED: " + ", ".join(missing))
    generated = oracle.OUTPUT / "reference/joint/oracle.f90"
    for source in [*reference["source"], reference["milly_observer_source"]]:
        raw = (ROOT / source["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source["source_sha256"]
        span = b"".join(raw.splitlines(keepends=True)[source["start"] - 1 : source["end"]])
        assert hashlib.sha256(span).hexdigest() == source["span_sha256"]
        if generated.is_file() and source is not reference["milly_observer_source"]:
            assert span in generated.read_bytes()
