import json
from dataclasses import replace

import jax
import numpy as np
import pytest

from jax_orchidee.driver import global_batch as batch
from jax_orchidee.driver import orchestration as driver


def lane(name, value=1.0):
    arrays = ({"x": np.asarray([value], dtype=np.float64), "optional": None},)
    args = batch.BlockArguments(
        arrays, np.ones((7, 48, 1)), np.arange(2, 9, dtype=np.int32), *({"x": np.asarray(value)} for _ in range(6))
    )
    return batch.PreparedLane(name, None, None, None, None, args, (("science", 1),))


def test_stack_all_eight_numeric_groups_and_keep_none():
    lanes = [lane("a", 1), lane("b", 2)]
    args = batch.stack_lanes(lanes)
    assert args.values_by_component[0]["optional"] is None
    np.testing.assert_array_equal(args.values_by_component[0]["x"], [[1], [2]])
    assert args.forcing_days.shape == (2, 7, 48, 1)
    assert args.science_day_numbers.shape == (7,)
    for i in range(3, 9):
        np.testing.assert_array_equal(args[i]["x"], [1, 2])


@pytest.mark.parametrize("change", ["static", "dtype", "none", "days", "shape"])
def test_incompatible_lanes_rejected(change):
    a, b = lane("a"), lane("b")
    args = b.arguments
    if change == "static":
        b = replace(b, static_signature=(("science", 2),))
    elif change == "dtype":
        b = replace(b, arguments=args._replace(forcing_days=args.forcing_days.astype(np.float32)))
    elif change == "none":
        b = replace(b, arguments=args._replace(values_by_component=({"x": np.ones(1), "optional": np.zeros(1)},)))
    elif change == "days":
        b = replace(b, arguments=args._replace(science_day_numbers=args.science_day_numbers + 7))
    elif change == "shape":
        b = replace(b, arguments=args._replace(forcing_days=np.ones((7, 48, 2))))
    with pytest.raises(ValueError):
        batch.stack_lanes([a, b])


def test_vmap_is_applied_to_callable_and_each_lane_is_independent(monkeypatch):
    def scalar(*args):
        return args[0][0]["x"] + args[3]["x"] + args[2][0], args[1].sum()

    monkeypatch.setattr(batch, "make_scalar_transition", lambda lane, **kw: scalar)
    lanes = [lane("a", 1.0), lane("b", 4.0)]
    fn = jax.jit(batch.make_batch_transition(lanes))
    x = fn(*batch.stack_lanes(lanes))
    np.testing.assert_array_equal(x[0], [[4], [10]])
    reversed_x = fn(*batch.stack_lanes(lanes[::-1]))
    np.testing.assert_array_equal(x[0][::-1], reversed_x[0])
    changed_x = fn(*batch.stack_lanes([lanes[0], lane("c", 8.0)]))
    np.testing.assert_array_equal(x[0][0], changed_x[0][0])
    assert not np.array_equal(x[0][1], changed_x[0][1])


def test_factory_retains_capture_validation():
    common = dict(
        context=None, initial_state=None, prebound_hydrol_runtime_static_tables=None, daily_carbon_dispatch={}
    )
    with pytest.raises(ValueError, match="boundary capture"):
        driver._paper_later_day_block_transition_factory("dummy", training_output_projector=lambda x: x, **common)
    with pytest.raises(ValueError, match="stable cache key"):
        driver._paper_later_day_block_transition_factory(
            "dummy", training_output_projector=lambda x: x, capture_pre_daily_training_boundaries=True, **common
        )


@pytest.mark.parametrize("case", ["matched", "unexpected", "one_sided", "inf", "wrong_shape"])
def test_numerical_gate_preserves_only_declared_sentinel(case):
    from scripts.dev.benchmark_global_batch import compare

    expected = {"rough": np.asarray([[np.nan, 2.0]], dtype=np.float64)}
    actual = {"rough": expected["rough"].copy()}
    contract = {"['rough']": {"inner_shape": [1, 2], "bare_soil_index": 0}}
    if case == "unexpected":
        actual["rough"][0, 1] = np.nan
    elif case == "one_sided":
        actual["rough"][0, 0] = 0
    elif case == "inf":
        actual["rough"][0, 0] = np.inf
    elif case == "wrong_shape":
        expected["rough"] = np.asarray([[np.nan], [2.0]])
        actual["rough"] = expected["rough"].copy()
    result = compare(actual, expected, sentinel_paths=contract)
    assert result["passed"] is (case == "matched")
    if case == "matched":
        assert result["allowed_sentinels"][0]["count"] == 1
        assert result["max_bound_ratio"] == 0


def test_stage_recovery_reuses_complete_reference_and_retries_only_incomplete(tmp_path):
    from scripts.dev.global_batch_stage_store import StageStore

    calls = []
    identity = {"inputs": "four-own-inputs", "runtime": "GPU-FP64", "science": "source"}
    store = StageStore(tmp_path, identity)

    def complete():
        calls.append("complete")
        return {"x": np.asarray([2.0])}, {"provenance": {"own_context_bound": True}}

    def interrupted():
        calls.append("interrupted")
        raise InterruptedError("bounded window ended")

    store.run("scalar_a", complete)
    with pytest.raises(InterruptedError):
        store.run("scalar_b", interrupted)
    recovered = StageStore(tmp_path, identity)
    ((value, record), reused) = recovered.run("scalar_a", complete)
    assert reused and record["provenance"]["own_context_bound"]
    np.testing.assert_array_equal(value["x"], [2.0])
    assert recovered.manifest["stages"]["scalar_b"]["status"] == "stopped"
    recovered.run("scalar_b", complete)
    assert calls == ["complete", "interrupted", "complete"]


@pytest.mark.parametrize("field", ["inputs", "science", "baseline", "runtime", "device", "compiler"])
def test_stage_recovery_rejects_contract_identity_change(tmp_path, field):
    from scripts.dev.global_batch_stage_store import StageStore

    identity = dict.fromkeys(("inputs", "science", "baseline", "runtime", "device", "compiler"), "original")
    store = StageStore(tmp_path, identity)
    store.run("B4_candidate", lambda: (np.asarray([1.0]), {}))
    with pytest.raises(ValueError, match="identity changed"):
        StageStore(tmp_path, {**identity, field: "changed"})


def test_stage_recovery_adopts_atomic_snapshot_after_manifest_interruption(tmp_path):
    from scripts.dev.global_batch_stage_store import StageStore

    identity = {"runtime": "same"}
    store = StageStore(tmp_path, identity)
    store.save("scalar_a", {"output": 3.0}, {"independent": True})
    manifest = json.loads(store.path.read_text())
    manifest["stages"]["scalar_a"] = {"status": "running"}
    store.path.write_text(json.dumps(manifest))
    (tmp_path / "stage_scalar_b.pkl.abandoned.tmp").write_bytes(b"partial pickle")
    recovered = StageStore(tmp_path, identity)
    value, record = recovered.load("scalar_a")
    assert value == {"output": 3.0} and record["independent"]
    assert not record["checkpoint_io"]["available"]
    assert recovered.manifest["stages"]["scalar_a"]["status"] == "complete"
    assert recovered.load("scalar_b") is None


def test_stage_recovery_rejects_corrupted_or_missing_complete_checkpoint(tmp_path):
    from scripts.dev.global_batch_stage_store import StageStore

    store = StageStore(tmp_path, {"runtime": "same"})
    store.save("scalar_a", 1.0, {})
    path = store.checkpoint_path("scalar_a")
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        store.load("scalar_a")
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        store.load("scalar_a")


def test_stage_checkpoint_io_records_actual_bytes_and_survives_resume(tmp_path):
    from scripts.dev.global_batch_stage_store import StageStore

    store = StageStore(tmp_path, {"inputs": "same"})
    store.run("B4_candidate", lambda: (np.arange(16, dtype=np.float64), {}))
    measured = store.manifest["stages"]["B4_candidate"]["checkpoint_io"]
    assert measured["available"]
    assert measured["bytes"] == store.checkpoint_path("B4_candidate").stat().st_size
    assert measured["serialization_seconds"] >= 0 and measured["atomic_write_seconds"] >= 0
    recovered = StageStore(tmp_path, {"inputs": "same"})
    assert recovered.load("B4_candidate")[1]["checkpoint_io"] == measured


def test_legacy_gpu_reference_requires_all_recorded_runtime_and_source_fields():
    from scripts.dev.benchmark_global_batch import validate_legacy_origin

    fields = ("python", "jax", "jaxlib", "numpy", "device_details", "x64_enabled", "baseline_ref")
    environment = dict.fromkeys(fields, "same")
    environment.update(git_head="origin", env=dict.fromkeys(
        ("XLA_FLAGS", "JAX_PLATFORMS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"), None))
    origin = {"environment": environment, "phases": [{"phase": "legacy_original_reference_saved"}]}
    evidence = validate_legacy_origin(origin, environment, scientific_diff="")
    assert "CUDA_driver_version" in evidence["missing_origin_metadata"]
    with pytest.raises(ValueError, match="jaxlib"):
        validate_legacy_origin(origin, {**environment, "jaxlib": "new"}, scientific_diff="")
    with pytest.raises(ValueError, match="source"):
        validate_legacy_origin(origin, environment, scientific_diff="jax_orchidee/changed.py")


def test_cache_policy_is_explicit_and_disables_old_cache(monkeypatch, tmp_path):
    from jax.experimental.compilation_cache import compilation_cache

    from scripts.dev.probe_global_batch_cache import configure_cache

    changes = {}
    resets = []
    monkeypatch.setattr(jax.config, "update", lambda key, value: changes.update({key: value}))
    monkeypatch.setattr(compilation_cache, "reset_cache", lambda: resets.append(dict(changes)))
    with pytest.raises(ValueError, match="explicit"):
        configure_cache("persistent", None)
    policy = configure_cache("persistent", tmp_path / "cache")
    assert policy["root"] == str((tmp_path / "cache").resolve())
    assert changes["jax_persistent_cache_min_compile_time_secs"] == 0
    configure_cache("off", None)
    assert changes["jax_enable_compilation_cache"] is False
    assert len(resets) == 2 and resets[0]["jax_compilation_cache_dir"] == policy["root"]


def test_reused_timing_keeps_actual_origin_policy_and_unknown_origin_explicit():
    from scripts.dev.benchmark_global_batch import record_timing_origin

    current = {"environment": {"cache_policy": "persistent", "cpu_affinity": [4, 5, 6, 7]}}
    previous = {"available": True, "environment": {"cache_policy": "disabled", "cpu_affinity": [0, 1, 2, 3]}}
    record_timing_origin(current, "B4_measurements", {"timing": {"warm": 3.0}, "execution_origin": previous}, reused=True)
    assert current["timing_origins"]["warm"]["reused_measurement"]
    assert current["timing_origins"]["warm"]["execution_origin"] == previous
    record_timing_origin(current, "B4_candidate", {"timing": {"old_compile": 600.0}}, reused=True)
    assert not current["timing_origins"]["old_compile"]["execution_origin"]["available"]


def test_recorder_preserves_exact_report_in_declared_window(monkeypatch, tmp_path):
    from scripts.dev.benchmark_global_batch import Recorder

    window = tmp_path / "windows" / "window001"
    monkeypatch.setenv("ORCHJAX_GLOBAL_BATCH_WINDOW_ID", str(window))
    recorder = Recorder(tmp_path / "model")
    try:
        recorder.result["status"] = "completed_stage_selection"
        recorder.flush()
        assert recorder.window_report == window / "benchmark-report.json"
        assert recorder.window_report.read_bytes() == (tmp_path / "model" / "report.json").read_bytes()
    finally:
        recorder.stop.set()


def test_semantic_reference_identity_ignores_pickle_aliasing_but_rejects_values():
    import copy
    from types import SimpleNamespace

    from scripts.dev.benchmark_global_batch import semantic_reference_identity

    first = lane("a", 2.0)
    args = SimpleNamespace(baseline_ref="pinned")
    # Separate clones retain exact content while losing shared object aliases.
    separate = replace(first, arguments=copy.deepcopy(first.arguments))
    original = semantic_reference_identity(args, first, first.arguments)
    assert original == semantic_reference_identity(args, separate, copy.deepcopy(first.arguments))
    assert original != semantic_reference_identity(args, lane("a", 3.0), first.arguments)
    assert original != semantic_reference_identity(SimpleNamespace(baseline_ref="new"), first, first.arguments)


@pytest.mark.parametrize("b1_mode", ["vmapped", "scalar"])
def test_staged_small_execution_preserves_candidate_and_each_independent_reference(monkeypatch, tmp_path, b1_mode):
    from types import SimpleNamespace

    from scripts.dev import benchmark_global_batch as harness

    lanes = [lane(point, float(i + 1)) for i, point in enumerate(harness.POINTS)]
    second = [item.arguments._replace(science_day_numbers=np.arange(9, 16, dtype=np.int32)) for item in lanes]
    compiled, bound = [], []
    interrupted = [False]

    def scalar(item, **kwargs):
        bound.append(item.landpoint_id)

        def transition(*args):
            updated = jax.tree_util.tree_map(lambda x: x + args[3]["x"], args[0])
            return updated, {"daily": args[1].sum() + args[2][0]}

        return transition

    def compile_small(fn, arguments, name, recorder):
        compiled.append(name)
        if name == "scalar_" + harness.POINTS[1] and not interrupted[0]:
            interrupted[0] = True
            raise InterruptedError("fake bounded stop during second independent compile")
        return jax.jit(fn)

    monkeypatch.setattr(batch, "make_scalar_transition", scalar)
    monkeypatch.setattr(harness, "lower_compile", compile_small)
    monkeypatch.setattr(harness, "legacy_factory", lambda baseline: None)
    monkeypatch.setattr(harness, "roughheight_sentinel_contract", lambda item: {})
    monkeypatch.setattr(harness, "stage_identity", lambda *args: {"small-test-contract": "fixed"})
    original = harness.run_two(jax.jit(scalar(lanes[0])), lanes[0].arguments, second[0])
    monkeypatch.setattr(harness, "import_legacy_reference", lambda *args: (jax.device_get(original), {"original": True}))
    args = SimpleNamespace(output_dir=tmp_path, baseline_ref="small-test", stage="all", warm_repeats=5, b1_mode=b1_mode)
    first = harness.Recorder(tmp_path)
    first.result["environment"] = {}
    try:
        with pytest.raises(InterruptedError):
            harness.staged_benchmark(args, first, lanes, second)
    finally:
        first.stop.set()
    manifest = json.loads((tmp_path / "stage_manifest.json").read_text())
    assert manifest["stages"]["B4_candidate"]["status"] == "complete"
    assert manifest["stages"]["scalar_" + harness.POINTS[0]]["status"] == "complete"
    assert manifest["stages"]["scalar_" + harness.POINTS[1]]["status"] == "stopped"
    resumed = harness.Recorder(tmp_path)
    resumed.result["environment"] = {}
    try:
        harness.staged_benchmark(args, resumed, lanes, second)
    finally:
        resumed.stop.set()
    assert resumed.result["all_required_stages_complete"]
    assert all(gate["passed"] for gate in resumed.result["gates"])
    assert compiled.count("B4_candidate") == 1
    assert compiled.count("scalar_" + harness.POINTS[0]) == 1
    assert set(harness.POINTS).issubset(bound)
    assert len(resumed.result["timing"]["B4_warm_frozen_state_core"]["samples_seconds"]) == 5
    b1_label = "B1_scalar_baseline" if b1_mode == "scalar" else "B1"
    assert len(resumed.result["timing"][b1_label + "_warm_frozen_state_core"]["samples_seconds"]) == 5
    if b1_mode == "scalar":
        assert "B1_validation" not in compiled and "B1_validation" not in resumed.result["required_stages"]
        assert "B1_scalar_baseline" in resumed.result["required_stages"]
    assert resumed.result["pooled_scalar"]["all_four_current_executable_gates_passed"]
    # A different pooled context must execute new gates/timings while retaining
    # all candidate and independently executed reference checkpoints.
    selected = harness.POINTS[1]
    args.pooled_scalar_point = selected
    changed = harness.Recorder(tmp_path)
    changed.result["environment"] = {}
    before = list(compiled)
    try:
        harness.staged_benchmark(args, changed, lanes, second)
    finally:
        changed.stop.set()
    assert changed.result["all_required_stages_complete"]
    assert changed.result["pooled_scalar"]["landpoint_id"] == selected
    assert changed.result["pooled_scalar"]["all_four_current_executable_gates_passed"]
    assert "B4_gates_pool_" + selected in changed.result["required_stages"]
    assert compiled.count("B4_candidate") == before.count("B4_candidate")
    assert compiled.count("scalar_" + selected) == before.count("scalar_" + selected)
    assert not changed.result["timing_origins"]["B4_warm_frozen_state_core"]["reused_measurement"]
