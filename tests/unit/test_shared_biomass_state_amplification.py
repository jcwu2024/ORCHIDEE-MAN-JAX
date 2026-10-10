"""Small CPU captures of genuine kernel operands; no network forward or AD."""
from collections import namedtuple
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_orchidee.stomate import carbon_kernels as kernels
from scripts.hpc.probe_shared_biomass_state_amplification import (
    BoundaryCollector, instrument_source_kernels, select_actual_sample, persist_capture,
    capture_tree, biomass_layout, timed_neural_component, validate_neural_outputs, install_turnover_overlay,
    observe_turnover_output_cotangent,
)


def captured_array(collector, metadata):
    return collector.arrays[metadata["array"]]


def test_capture_real_npp_maintenance_denominator_and_restore_aliases(monkeypatch):
    import jax_orchidee.stomate.integration as integration
    original, original_alias = kernels.npp_closed_update, integration.npp_closed_update
    collector = BoundaryCollector()
    biomass = jnp.ones((1, 2, 12, 1), dtype=jnp.float64)
    maintenance = jnp.zeros((1, 2, 12), dtype=jnp.float64).at[0, 1, 0].set(1e-139).at[0, 1, 1].set(2e-139)
    kwargs = dict(biomass=biomass, gpp=jnp.zeros((1, 2), dtype=jnp.float64),
        f_alloc=jnp.zeros((1, 2, 12), dtype=jnp.float64), resp_maint_part=maintenance,
        pft_present=jnp.array([[False, True]]), frac_growthresp=jnp.array([.28, .28]))
    with jax.disable_jit(), instrument_source_kernels(collector, kernels):
        assert integration.npp_closed_update is kernels.npp_closed_update
        actual = integration.npp_closed_update(**kwargs)
    assert kernels.npp_closed_update is original and integration.npp_closed_update is original_alias
    expected = original(**kwargs)
    for left, right in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
        np.testing.assert_array_equal(left, right)
    ratio = next(record for record in collector.records if record["symbol"] == "_stable_ratio_for_ad")
    denom = captured_array(collector, ratio["inputs"]["denominator"])
    np.testing.assert_allclose(denom[0, 1, 0], 3e-139, rtol=1e-15, atol=0)
    assert ratio["parent"] == 0 and collector.records[0]["symbol"] == "npp_closed_update"
    assert captured_array(collector, ratio["ratio_branch_diagnostics"]["tangent_representable"])[0, 1, 0]
    assert any(record["symbol"] == "jnp.where" for record in collector.records)


def test_capture_turnover_uses_postturnover_mass_not_initial_mass():
    collector = BoundaryCollector()
    initial_mass = 1e-139
    biomass = jnp.zeros((1, 2, 12, 1), dtype=jnp.float64).at[0, 1, 0, 0].set(initial_mass)
    kwargs = dict(biomass=biomass, turnover=jnp.zeros_like(biomass),
        leaf_age=jnp.full((1, 2, 4), 1000., dtype=jnp.float64),
        leaf_frac=jnp.full((1, 2, 4), .25, dtype=jnp.float64),
        t2m_longterm=jnp.array([293.15]), is_tree=jnp.array([False, True]),
        natural=jnp.array([False, True]), ok_laidev=jnp.array([False, False]),
        leafagecrit=jnp.array([100., 100.]))
    with jax.disable_jit(), instrument_source_kernels(collector, kernels):
        result = kernels.turnover_leaf_age_fall(**kwargs)
    ratio = next(record for record in collector.records if record["symbol"] == "_stable_ratio_for_ad")
    denominator = captured_array(collector, ratio["inputs"]["denominator"])
    np.testing.assert_array_equal(denominator[0, 1, 0], result[0][0, 1, 0, 0])
    assert 0 < denominator[0, 1, 0] < initial_mass
    assert any(record["symbol"] == "jnp.minimum" for record in collector.records)
    where = next(record for record in collector.records if record["symbol"] == "jnp.where")
    assert captured_array(collector, where["inputs"]["condition"]).dtype == np.bool_


def test_instrument_restore_on_exception_and_owned_payload():
    original = kernels.turnover_leaf_age_fall
    collector = BoundaryCollector()
    with pytest.raises(ValueError):
        with instrument_source_kernels(collector, kernels):
            raise ValueError("intentional boundary failure")
    assert kernels.turnover_leaf_age_fall is original
    value = np.array([1., 2.])
    metadata = collector.put("real", value)
    value[:] = 99
    np.testing.assert_array_equal(captured_array(collector, metadata), [1, 2])


def test_native_turnover_output_cotangent_observation_restores_actual_symbol():
    original = kernels.turnover_leaf_age_fall
    captured = []
    mass = jnp.zeros((1, 2, 12, 1), dtype=jnp.float64).at[0,1,0,0].set(2.)
    kwargs = dict(turnover=jnp.zeros_like(mass), leaf_age=jnp.zeros((1,2,4)),
        leaf_frac=jnp.zeros((1,2,4)).at[0,1,0].set(1.), t2m_longterm=jnp.array([293.15]),
        is_tree=jnp.array([False,True]), natural=jnp.array([False,True]),
        ok_laidev=jnp.array([False,False]), leafagecrit=jnp.array([100.,100.]))
    def loss(value):
        return jnp.sum(kernels.turnover_leaf_age_fall(biomass=value, **kwargs)[0])
    baseline = jax.grad(loss)(mass)
    with jax.disable_jit(), observe_turnover_output_cotangent(kernels, captured.append):
        observed = jax.grad(loss)(mass)
    np.testing.assert_array_equal(observed, baseline)
    assert kernels.turnover_leaf_age_fall is original and len(captured) == 1
    np.testing.assert_array_equal(captured[0][0], np.ones_like(mass))
    np.testing.assert_array_equal(captured[0][2], np.zeros((1,2,4)))


def test_consolidated_npz_and_metadata_are_small_and_restore_exact_arrays(tmp_path):
    import json
    collector = BoundaryCollector()
    collector.put("call0.input.denominator", np.array([5.805124529028718e-139]))
    collector.records.append({"symbol": "local_actual_boundary"})
    path = tmp_path/"capture.json"
    persist_capture(path, {"status": "captured", "GPU_executed": False}, collector)
    report = json.loads(path.read_text())
    assert report["capture_array_bytes"] == 8 and report["GPU_executed"] is False
    with np.load(path.with_suffix(".npz")) as arrays:
        np.testing.assert_array_equal(arrays["call0.input.denominator"], [5.805124529028718e-139])


def test_actual_scalar_row_and_micro4_binding_use_artifact275():
    Inputs = namedtuple("Inputs", "continuous_state")
    Sample = namedtuple("Sample", "inputs")
    states742 = np.arange(32*3854, dtype=np.float64).reshape(32, 3854)
    states275 = states742+1e6
    bundle = {"artifact_ids": (742, 275), "payloads": [{"daily": Sample(Inputs(states742))}, {"daily": Sample(Inputs(states275))}]}
    sample, identity = select_actual_sample(jax, bundle, 275, 17)
    np.testing.assert_array_equal(sample.inputs.continuous_state, states275[17])
    assert identity["artifact_index"] == 1 and identity["micro4_row_start"] == 16 and identity["micro4_row_stop"] == 20
    with pytest.raises(ValueError): select_actual_sample(jax, bundle, 999, 17)


def test_capture_rejects_tracers_and_excess_diagnostic_memory():
    collector = BoundaryCollector(max_bytes=8)
    collector.put("one", np.ones(1))
    with pytest.raises(MemoryError): collector.put("two", np.ones(1))
    FakeTracer = type("FakeTracer", (), {"__module__": "jax.fake"})
    with pytest.raises(RuntimeError, match="Tracer"): BoundaryCollector().put("trace", FakeTracer())


def test_public_next_state_and_defined_masks_remain_consolidated():
    Groups = namedtuple("Groups", "other")
    collector = BoundaryCollector()
    actual = {"predicted": Groups(np.array([1e-139, 2.])),
              "expected_defined": Groups(np.array([True, False]))}
    payload = capture_tree(collector, "actual_next_state_binding", actual)
    paths = {entry["path"]: entry for entry in payload["leaves"]}
    np.testing.assert_array_equal(captured_array(collector, paths["['predicted'].other"]), actual["predicted"].other)
    np.testing.assert_array_equal(captured_array(collector, paths["['expected_defined'].other"]), [True, False])
    assert all("array" in entry and "values" not in entry for entry in payload["leaves"])


def test_actual_contract_biomass_mapping_and_minimum_tie():
    leaf = SimpleNamespace(key="stomate.biomass", start=2231, stop=2255, shape=(2, 12, 1),
        axis_names=("nvm", "nparts", "nelements"), selected_pft_indices=(0, 13), source_ref="actual source")
    layout = biomass_layout(SimpleNamespace(state_leaves=[leaf]))
    assert layout["local_coordinate"] == 12 and layout["unravelled_coordinate"] == [1, 0, 0]
    collector = BoundaryCollector()
    def actual_minimum(x, y): return np.minimum(x, y)
    output = collector.wrap("jnp.minimum", actual_minimum)(np.array([1., 2.]), np.array([1., 3.]))
    np.testing.assert_array_equal(output, [1., 2.])
    np.testing.assert_array_equal(captured_array(collector, collector.records[0]["comparison_diagnostics"]["tie"]), [True, False])


def test_forward_component_jit_under_eager_outer_context_preserves_values():
    events, captured = [], []
    def function(parameters, inputs, *, factor): return parameters*inputs+factor
    observed = timed_neural_component(function, "small_neural", "component_jit",
        lambda event, **metadata: events.append(event), capture=captured.append)
    with jax.disable_jit():
        actual = observed(jnp.array([2., 3.]), jnp.array([4., 5.]), factor=2.)
        second = observed(jnp.array([3., 4.]), jnp.array([4., 5.]), factor=2.)
    np.testing.assert_array_equal(actual, [10., 17.])
    np.testing.assert_array_equal(second, [14., 22.])
    assert events.count("small_neural_compiled") == 1 and len(captured) == 2
    reused_events = []
    reused = timed_neural_component(function, "small_neural", "component_jit",
        lambda event, **metadata: reused_events.append(event))
    with jax.disable_jit():
        np.testing.assert_array_equal(reused(jnp.array([4., 5.]), jnp.array([4., 5.]), factor=2.), [18., 27.])
    assert "small_neural_same_process_executable_reused" in reused_events
    assert "small_neural_compiled" not in reused_events


def test_actual_neural_cache_shape_dtype_guard():
    names = "source_tokens source_mask source_family_index expert_axis_tokens expert_axis_mask pft_tokens pft_mask process_tokens process_source_mask"
    Context = namedtuple("Context", names)
    context = Context(np.zeros((3, 4), "float32"), np.ones(3, bool), np.zeros(3, "int32"),
        np.zeros((1, 2, 4), "float32"), np.ones((1, 2), bool), np.zeros((2, 4), "float32"),
        np.ones(2, bool), np.zeros((1, 4), "float32"), np.ones((1, 3), bool))
    head = SimpleNamespace(learned_schema=SimpleNamespace(fields=[SimpleNamespace(name="actual", shape=(2,))]),
        shared_config=SimpleNamespace(width=4), layout=SimpleNamespace(field_expert_index=[0], active_pft_indices=[0, 13]))
    outputs = {"actual_neural_context": context,
        "actual_neural_raw_outputs": ({"actual": np.zeros(2, "float32")}, {"actual": np.ones(2, bool)})}
    assert validate_neural_outputs(outputs, head)["raw_shape_dtype_finite"]
    outputs["actual_neural_raw_outputs"][0]["actual"] = np.zeros(2, "float64")
    with pytest.raises(ValueError, match="shape/dtype/finite"):
        validate_neural_outputs(outputs, head)


def test_verified_single_function_overlay_restores_alias_and_capture_globals(tmp_path):
    import hashlib
    import inspect
    original = kernels.turnover_leaf_age_fall
    source = inspect.getsource(original)
    path = tmp_path/"guarded.py"
    path.write_text(source)
    digest = hashlib.sha256(source.encode()).hexdigest()
    with pytest.raises(ValueError, match="hash differs"):
        with install_turnover_overlay(kernels, path, "wrong"): pass
    with install_turnover_overlay(kernels, path, digest) as metadata:
        assert kernels.turnover_leaf_age_fall is not original and metadata["checkout_edited"] is False
        collector = BoundaryCollector()
        with instrument_source_kernels(collector, kernels):
            # The overlay's globals are outside a registered module but its
            # original ratio alias must still be observed by instrumentation.
            values = jnp.zeros((1, 2, 12, 1), dtype=jnp.float64)
            kernels.turnover_leaf_age_fall(values, values, jnp.zeros((1, 2, 4)), jnp.zeros((1, 2, 4)),
                jnp.array([293.15]), is_tree=jnp.array([False, True]), natural=jnp.array([False, True]),
                ok_laidev=jnp.array([False, False]), leafagecrit=jnp.array([100., 100.]))
        assert any(row["symbol"] == "_stable_ratio_for_ad" for row in collector.records)
    assert kernels.turnover_leaf_age_fall is original
