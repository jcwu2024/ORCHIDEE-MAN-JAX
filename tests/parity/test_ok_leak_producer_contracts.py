"""Bounded C6/C9 producer regressions, without data files or a model day.

Fortran authority: stomate.f90:3380-3388 supplies dt_sechiba/one_day;
stomate_soilcarbon.f90:1212-1234 applies dt only to dry deposition and
:1495-1525 carries canopy storage; constantes_var.f90:294 fixes sro_bottom=5.
The layer-six split follows stomate_soilcarbon.f90:1711-1785 (l>sro_bottom).
Producer calls and real small components are exercised below. The full
OK_LEAK carbon core is replaced, so these are not full compiled-core parity.
"""

from types import SimpleNamespace
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from jax_orchidee import coupled
from jax_orchidee.coupled import StomateOkLeakBoundaryInputs
from jax_orchidee.driver import orchestration as driver
from jax_orchidee.stomate.soilcarbon_kernels import (
    soilcarbon_leak_decompose_poc_doc,
    soilcarbon_leak_tf_doc_ground_fluxes,
)


class ComponentProbe(NamedTuple):
    interception_storage: object
    dry: object
    wet_canopy: object
    wet_ground_input: object
    wet_ground: object
    wet_flood: object
    soilwater: object
    cutoff: object


@pytest.fixture(autouse=True)
def _isolate_unrelated_permafrost_activity(monkeypatch):
    # The target is TF-DOC and runoff-cutoff producer wiring. Do not compile
    # or claim admission of the distinct permafrost moisture/deep-grid path.
    monkeypatch.setattr(
        coupled, "stomate_permafrost_activity_ok_leak_inputs",
        lambda **args: SimpleNamespace(ok_leak_inputs={"tprof": args["tdeep"]}, output_inputs={}, provenance=()),
    )
    monkeypatch.setattr(
        driver, "stomate_permafrost_decomposition_controls",
        lambda *args, **kwargs: SimpleNamespace(prmfrst_soilc_tempctrl=jnp.ones((1, 32, 2)),
                                               prmfrst_soilc_tempctrl_doc=jnp.ones((1, 32, 2))),
    )


def _fixture(*, tf_on=True, cutoff=None):
    """Synthetic positive-leaf/cover fixture with production 11/32 dimensions."""
    nvm, nslm, ndeep = 2, 11, 32
    veget = np.array([[0.0, 0.4]])
    biomass = np.zeros((1, nvm, 12, 1))
    biomass[0, 1, 0, 0] = 100.0
    state = {name: np.zeros((1,)) for name in driver._OK_LEAK_HALF_HOUR_STATE_FIELDS}
    state.update(
        biomass=biomass, veget_max=veget, sla_calc=np.full((1, nvm), 0.02),
        turnover_daily=np.zeros_like(biomass), bm_to_litter=np.zeros_like(biomass),
        interception_storage=np.array([[[0.0], [0.004]]]),
        altmax=np.ones((1, nvm)), fixed_cryoturbation_depth=np.ones((1, nvm)),
        carbon_32l=np.zeros((1, 3, nvm, ndeep)),
    )
    payload = dict(
        # Independent five-layer water expectation: sum(k*k,k=1..5)=55.
        # Layer six and below are deliberately large to expose nslm leakage.
        soil_mc=np.array([[[float(k)] for k in range(1, nslm + 1)]]),
        wat_flux=np.zeros((1, nslm, 1)), runoff_per_soil=np.zeros((1, 1)),
        drainage_per_soil=np.zeros((1, 1)), runoff2peat=np.zeros((1,)),
        canopy2ground=np.array([[0.0, 0.05]]), precip2ground=np.array([[0.0, 0.2]]),
        precip2canopy=np.array([[0.0, 0.1]]), temp_sol=np.array([280.0]),
        tdeep=np.full((1, ndeep, nvm), 280.0), hsdeep=np.ones((1, ndeep, nvm)),
        shumdiag_peat=np.ones((1, nslm)),
    )
    base_inputs = dict(
        nslm=nslm, ndeep=ndeep,
        z_soil=np.concatenate(([0.0], np.cumsum(np.arange(1, nslm + 1)))),
        zi_soil=np.arange(ndeep, dtype=float) + 0.5,
        zf_soil_b=np.arange(ndeep + 1, dtype=float),
        poor_soils=np.zeros(1), flood_frac=np.array([0.25]),
        bulk_dens=np.array([1650.0]), clay=np.array([0.3]), flux_red=np.ones(1),
        interception_storage=state["interception_storage"],
        carbon_32l=state["carbon_32l"],
    )
    if cutoff is not None:
        base_inputs["sro_bottom"] = cutoff
    return SimpleNamespace(
        state=state, payload=payload,
        initial=SimpleNamespace(fields_by_component={
            "slowproc_stomate_previous_step_state": state,
            "hydrol_previous_step_state": payload,
            "thermosoil_previous_step_state": {"tdeep": payload["tdeep"], "hsdeep": payload["hsdeep"]},
        }),
        base=StomateOkLeakBoundaryInputs(ok_leak_inputs=base_inputs, output_inputs={}, provenance=()),
        runtime=driver.DriverRuntimeScalars(1800.0, 86400.0, True, False),
        values={"TF_DOC": "TRUE" if tf_on else "FALSE", "CONC_DOC_RAIN": "3.02",
                "PERMA_PEAT": "FALSE", "frozen_respiration_func": "1", "MOIST_FUNC_MOYANO": "FALSE"},
        trees=np.array([False, True]), peat=np.array([False, False]),
        rprof=np.ones((1, nvm)), pref=np.array([1, 1], dtype=np.int32),
    )


def _component_core(**args):
    """Replace only full carbon core; run source-backed TF ground/storage."""
    ground = soilcarbon_leak_tf_doc_ground_fluxes(
        args["doc_precip2ground"], args["doc_precip2canopy"], args["dry_dep_canopy"],
        args["interception_storage"], args["canopy2ground"], args["veget_max"], args["flood_frac"],
    )
    return ComponentProbe(
        ground.interception_storage, args["dry_dep_canopy"], args["doc_precip2canopy"],
        args["doc_precip2ground"], ground.wet_dep_ground, ground.wet_dep_flood,
        args["soilwater_31mm"], jnp.asarray(args.get("sro_bottom", 5)),
    )


def _install_component_seam(monkeypatch, fixture):
    records = []

    def core(**args):
        records.append(args)
        return _component_core(**args)

    def updates(result):
        values = {name: fixture.state[name] for name in driver._OK_LEAK_HALF_HOUR_STATE_FIELDS}
        values["interception_storage"] = result.interception_storage
        return values

    monkeypatch.setattr(driver, "stomate_ok_leak_explicit", core)
    monkeypatch.setattr(driver, "_paper_half_hour_ok_leak_state_updates", updates)
    return records


def _ordinary(fixture, *, compiled=False, steps=2):
    return driver._paper_half_hour_ok_leak_fold_from_entries(
        entry_payloads=tuple(dict(fixture.payload) for _ in range(steps)),
        maintenance_resp_parts=np.zeros((steps, 1, 2, 12)), initial_state=fixture.initial,
        pre_step_boundary=fixture.base, runtime=fixture.runtime, run_def_values=fixture.values,
        rprof=fixture.rprof, pref_soil_veg=fixture.pref, is_tree=fixture.trees,
        is_peat=fixture.peat, dayno=2, use_compiled_ok_leak=compiled,
    )


def _initialization(fixture):
    return driver._paper_ok_leak_boundary_from_sources(
        base=fixture.base, previous_state=fixture.initial, runtime=fixture.runtime,
        run_def_values=fixture.values,
        daily_carbon=SimpleNamespace(post_npp=SimpleNamespace(turnover=SimpleNamespace(biomass=fixture.state["biomass"]))),
        bundles=SimpleNamespace(maintenance_inputs={"rprof": fixture.rprof},
                                alloc_inputs={"sla_calc": fixture.state["sla_calc"]}),
        final_entry_payload=fixture.payload, veget_max=fixture.state["veget_max"],
        pref_soil_veg=fixture.pref, is_tree=fixture.trees, is_peat=fixture.peat,
        restart_altmax=fixture.state["altmax"],
        restart_fixed_cryoturbation_depth=fixture.state["fixed_cryoturbation_depth"],
        npts=1, dayno=2,
    )[0]


@pytest.mark.parametrize("tf_on", [True, False])
def test_c6_ordinary_actual_producer_real_tf_and_two_step_storage(monkeypatch, tf_on):
    f = _fixture(tf_on=tf_on)
    records = _install_component_seam(monkeypatch, f)
    result, _ = _ordinary(f)
    # Independent scalar transcription of source1218-1225/1495-1525.
    dry = 0.00092 * (1800.0 / 86400.0) * 100.0 * 0.4 if tf_on else 0.0
    wet_canopy = 0.1 * 3.02e-3 if tf_on else 0.0
    wet_ground = 0.2 * 3.02e-3 if tf_on else 0.0
    old = 0.004
    for args in records:
        released = min(0.05 * 100.0e-3, old)
        np.testing.assert_allclose(args["dry_dep_canopy"][0, 1, 0], dry, rtol=1e-13)
        np.testing.assert_allclose(args["doc_precip2canopy"][0, 1, 0], wet_canopy, rtol=1e-13)
        np.testing.assert_allclose(args["doc_precip2ground"][0, 1, 0], wet_ground, rtol=1e-13)
        old += dry + wet_canopy - released
    np.testing.assert_allclose(result.interception_storage[0, 1, 0], old, atol=1e-15)


def test_c6_initialization_actual_producer_uses_sechiba_clock():
    args = _initialization(_fixture()).ok_leak_inputs
    np.testing.assert_allclose(args["dry_dep_canopy"][0, 1, 0], 0.00092 / 48.0 * 100.0 * 0.4)
    # Rain input already is a step mass; no additional 1/48.
    np.testing.assert_allclose(args["doc_precip2ground"][0, 1, 0], 0.2 * 3.02e-3)


def _capture_compiled_forward(monkeypatch, fixture):
    captured = {}

    def compiled(**kwargs):
        captured.update(kwargs)
        return kwargs["initial"], None, None

    monkeypatch.setattr(driver, "_paper_compiled_ok_leak_fold_jit", compiled)
    _ordinary(fixture, compiled=True, steps=1)
    return captured


def test_c6_actual_compiled_producer_forwards_half_hour_clock(monkeypatch):
    args = _capture_compiled_forward(monkeypatch, _fixture())
    assert args["tf_doc_dt_days"] == 1800.0 / 86400.0


def _compiled_body(monkeypatch, *, tf_on=True, cutoff=None):
    f = _fixture(tf_on=tf_on, cutoff=cutoff)
    _install_component_seam(monkeypatch, f)
    # Exercise the actual scan/producer body with a TF/storage-only core seam.
    # This is a narrow scan compilation, never full scientific OK_LEAK parity.
    captured = _capture_compiled_forward(monkeypatch, f)
    captured["tf_doc_dt_days"] = 1800.0 / 86400.0
    return f, driver._paper_compiled_ok_leak_fold(**captured)[1]


@pytest.mark.parametrize("tf_on", [True, False])
def test_c6_real_compiled_body_tf_storage_component(monkeypatch, tf_on):
    _, result = _compiled_body(monkeypatch, tf_on=tf_on)
    dry = 0.00092 / 48.0 * 100.0 * 0.4 if tf_on else 0.0
    wet_canopy = 0.1 * 3.02e-3 if tf_on else 0.0
    np.testing.assert_allclose(result.dry[0, 1, 0], dry)
    np.testing.assert_allclose(result.wet_canopy[0, 1, 0], wet_canopy)
    np.testing.assert_allclose(result.interception_storage[0, 1, 0], dry + wet_canopy, atol=1e-15)
    expected_wet = (0.2 * 3.02e-3 if tf_on else 0.0) + 0.004
    np.testing.assert_allclose(result.wet_ground[0, 1, 0], expected_wet * 0.75)
    np.testing.assert_allclose(result.wet_flood[0, 1, 0], expected_wet * 0.25)


@pytest.mark.parametrize("mode", ["ordinary", "initialization", "compiled"])
@pytest.mark.parametrize("cutoff", [None, 3])
def test_c9_actual_producers_use_one_explicit_cutoff(monkeypatch, mode, cutoff):
    f = _fixture(cutoff=cutoff)
    expected_cutoff = 5 if cutoff is None else cutoff
    if mode == "initialization":
        args = _initialization(f).ok_leak_inputs
        water, forwarded = args["soilwater_31mm"], args.get("sro_bottom", 5)
    elif mode == "ordinary":
        records = _install_component_seam(monkeypatch, f)
        _ordinary(f, steps=1)
        water, forwarded = records[0]["soilwater_31mm"], records[0].get("sro_bottom", 5)
    else:
        _, result = _compiled_body(monkeypatch, cutoff=cutoff)
        water, forwarded = result.soilwater, result.cutoff
    # Direct source weighted sum, not a second call to the water helper.
    expected_water = sum(k * k for k in range(1, expected_cutoff + 1))
    np.testing.assert_allclose(water, [[expected_water]], rtol=0.0, atol=1e-12)
    assert int(forwarded) == expected_cutoff


def test_c9_actual_compiled_producer_forwards_explicit_api_cutoff(monkeypatch):
    args = _capture_compiled_forward(monkeypatch, _fixture(cutoff=3))
    assert args["sro_bottom"] == 3
    assert args["nslm"] == 11
    assert args["ndeep"] == 32


@pytest.mark.parametrize("mode", ["ordinary", "compiled"])
def test_c9_producer_cutoff_reaches_source_layer_six_split(monkeypatch, mode):
    f = _fixture()
    if mode == "ordinary":
        records = _install_component_seam(monkeypatch, f)
        _ordinary(f, steps=1)
        cutoff = records[0].get("sro_bottom", 5)
        # Require water/core agreement before interpreting the split.
        np.testing.assert_allclose(records[0]["soilwater_31mm"], [[55.0]])
    else:
        _, result = _compiled_body(monkeypatch)
        cutoff = int(result.cutoff)
    carbon = np.zeros((1, 3, 2, 32))
    carbon[0, 0, 1, 4:6] = 30.0
    result = soilcarbon_leak_decompose_poc_doc(
        carbon, np.zeros((1, 2, 32, 2, 7, 1)), np.zeros((1, 2, 32, 3)),
        np.zeros((1, 2)), np.zeros((1, 2, 32)), np.zeros((1, 32, 2, 7)),
        np.ones((1, 32, 2, 3)), np.zeros((1, 3, 3)), np.ones((1, 32, 1)),
        np.array([0, 0]), np.array([0.6]), np.array([True, True]),
        np.array([False, False]), np.array([False, False]), np.zeros(1),
        dt_days=1.0 / 48.0, priming=False, sro_bottom=cutoff,
    )
    # Source: dry=dt*30*(1-.6)=.25, flood=dt*30*.6/3=.125,
    # CUE=.3; flood is retained as free DOC only below layer5.
    # Pool IACT is one-based5, Python4 (constantes_var pool ordering).
    np.testing.assert_allclose(result.doc[0, 1, 4:6, 0, 4, 0], [0.075, 0.1125], atol=1e-14)
