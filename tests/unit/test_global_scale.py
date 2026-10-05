"""Small real JAX evidence for full-lane scale comparison and checkpoint isolation."""

import hashlib
import io
import json
import pickle
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from jax_orchidee.driver import global_batch as batch
from scripts.dev import benchmark_global_scale as scale
from scripts.dev import prepare_global_scale_remote as remote
from scripts.dev.global_batch_stage_store import StageStore


def lane(name, value):
    arguments = batch.BlockArguments(
        ({"x": np.asarray([value], dtype=np.float64), "optional": None},),
        np.ones((7, 48, 1)), np.arange(2, 9, dtype=np.int32),
        *({"x": np.asarray(value)} for _ in range(6)),
    )
    return batch.PreparedLane(name, None, None, None, None, arguments, (("science", 1),))


def prepared(count=64):
    ids = list(scale.base.POINTS) + [f"point_{index:03d}" for index in range(4, count)]
    lanes = [lane(point, float(index + 1)) for index, point in enumerate(ids)]
    # Include discrete state and preserve an optional None leaf across both blocks.
    lanes = [replace(item, arguments=item.arguments._replace(values_by_component=(
        {**item.arguments.values_by_component[0], "mask": np.asarray([True], dtype=bool)},))) for item in lanes]
    second = [item.arguments._replace(science_day_numbers=np.arange(9, 16, dtype=np.int32)) for item in lanes]
    return lanes, second


def scalar_transition(*args):
    updated = jax.tree_util.tree_map(lambda value: value if value.dtype.kind == "b" else value + args[3]["x"], args[0])
    return updated, {"daily": args[1].sum() + args[2][0], "mask": args[0][0]["mask"]}


def mock_science(monkeypatch):
    monkeypatch.setattr(scale.base, "roughheight_sentinel_contract", lambda item: {})
    monkeypatch.setattr(scale.base, "legacy_factory", lambda ref: None)
    monkeypatch.setattr(batch, "make_scalar_transition", lambda lane, **kwargs: scalar_transition)


def test_scale_all_unique_lanes_and_same_workload_baseline_with_B64_failure(monkeypatch, tmp_path):
    mock_science(monkeypatch)
    lanes, second = prepared()
    anchors = [scale.base.run_two(jax.jit(scalar_transition), item.arguments, other)
               for item, other in zip(lanes[:4], second[:4], strict=True)]
    b4 = jax.tree_util.tree_map(lambda *values: np.stack(values), *anchors)
    values = {"B4_batch_only_v1": b4, **{"scalar_" + point: value for point, value in zip(scale.base.POINTS, anchors)}}
    monkeypatch.setattr(scale, "load_anchor_references", lambda *args: (values, {"manifest_sha256": "sealed"}))
    monkeypatch.setattr(scale.base, "stage_identity", lambda *args: {"science": "same"})
    compiled, calls, interrupted = [], [], [False]

    def compile_small(fn, arguments, name, recorder):
        compiled.append(name)
        if name == "B64_candidate" and not interrupted[0]:
            interrupted[0] = True
            saved = json.loads((tmp_path / "stage_manifest.json").read_text())
            assert saved["stages"]["B16_scale"]["status"] == "complete"
            raise InterruptedError("budget stop before B64 compile completes")
        executable = jax.jit(fn)

        def tracked(*arguments):
            if name == "B4_resident":
                calls.append(tuple(np.asarray(arguments[3]["x"]).tolist()))
            return executable(*arguments)

        return tracked

    monkeypatch.setattr(scale.base, "lower_compile", compile_small)
    args = SimpleNamespace(output_dir=tmp_path, sizes=[16], warm_repeats=5, baseline_ref="small")
    first = scale.base.Recorder(tmp_path)
    first.result["environment"] = {}
    try:
        scale.scale_benchmark(args, first, lanes, second)
    finally:
        first.stop.set()
    assert first.result["all_required_stages_complete"]
    args.sizes = [16, 64]
    failed = scale.base.Recorder(tmp_path)
    failed.result["environment"] = {}
    try:
        with pytest.raises(InterruptedError):
            scale.scale_benchmark(args, failed, lanes, second)
    finally:
        failed.stop.set()
    manifest = json.loads((tmp_path / "stage_manifest.json").read_text())
    assert compiled == ["B4_resident", "B16_candidate", "B4_resident", "B64_candidate"]
    assert manifest["stages"]["B16_scale"]["status"] == "complete"
    assert manifest["stages"]["B64_scale"]["status"] == "stopped"
    first_bytes = (tmp_path / "stage_B16_scale.pkl").read_bytes()
    second_run = scale.base.Recorder(tmp_path)
    second_run.result["environment"] = {}
    try:
        scale.scale_benchmark(args, second_run, lanes, second)
    finally:
        second_run.stop.set()
    identity = {"science": "same", "contract": "unique-real-points-two-seven-day-blocks-B16-B64-v1",
                "reference_manifest_sha256": "sealed"}
    store = StageStore(tmp_path, identity)
    assert second_run.result["all_required_stages_complete"]
    assert (tmp_path / "stage_B16_scale.pkl").read_bytes() == first_bytes
    assert compiled.count("B16_candidate") == 1 and compiled.count("B4_resident") == 3
    assert second_run.result["timing_origins"]["B16_warm_frozen_state_core"]["reused_measurement"]
    assert not second_run.result["timing_origins"]["B64_warm_frozen_state_core"]["reused_measurement"]
    for size in (16, 64):
        saved, record = store.load(f"B{size}_scale")
        assert saved["outputs"][0][0][0]["x"].shape == (size, 1)
        assert saved["outputs"][0][0][0]["mask"].dtype == np.dtype("bool")
        assert record["provenance"]["point_ids"] == [item.landpoint_id for item in lanes[:size]]
        for key in (f"B{size}_warm_frozen_state_core", f"B{size}_B4_group_serial"):
            assert saved["measurements"][key]["valid_point_days"] == size * 14
            assert len(saved["measurements"][key]["samples_seconds"]) == 5
        assert all(gate["passed"] for gate in record["gates"])
        names = {gate["name"] for gate in record["gates"]}
        assert f"B{size}_cross_group_permutation" in names
        assert f"B{size}_replacement_all_lanes_isolation" in names
    # Every group really ran its own parameter/state inputs, including points 61-64.
    assert tuple(range(61, 65)) in calls and tuple(range(5, 9)) in calls


@pytest.mark.parametrize("invalid", ["duplicate", "anchor", "static", "float32", "nonfinite", "day", "forcing"])
def test_scale_input_screen_rejects_scientific_or_shape_changes(monkeypatch, invalid):
    mock_science(monkeypatch)
    lanes, second = prepared(16)
    item = lanes[-1]
    if invalid == "duplicate":
        lanes[-1] = replace(item, landpoint_id=lanes[0].landpoint_id)
    elif invalid == "anchor":
        lanes[:2] = lanes[:2][::-1]
    elif invalid == "static":
        lanes[-1] = replace(item, static_signature=(("science", 2),))
    elif invalid == "float32":
        lanes = [replace(value, arguments=value.arguments._replace(forcing_days=value.arguments.forcing_days.astype(np.float32)))
                 for value in lanes]
        second = [value._replace(forcing_days=value.forcing_days.astype(np.float32)) for value in second]
    elif invalid == "nonfinite":
        item.arguments.values_by_component[0]["x"][0] = np.nan
    elif invalid == "day":
        second = [value._replace(science_day_numbers=value.science_day_numbers + 1) for value in second]
    else:
        lanes = [replace(value, arguments=value.arguments._replace(forcing_days=value.arguments.forcing_days[:, :47]))
                 for value in lanes]
        second = [value._replace(forcing_days=value.forcing_days[:, :47]) for value in second]
    with pytest.raises(ValueError):
        scale.screen_inputs(lanes, second)


def test_memory_admission_uses_physical_free_slack_actual_trees_and_compiler_temps():
    inputs = (np.ones((64, 8), dtype=np.float64),)
    output = (np.ones((4, 3), dtype=np.float64),)
    memory = {4: {"available": True, "temp_size_in_bytes": 100_000}, 16: {"available": False}}
    enough = {"applicable": True, "total_bytes": 32 * 1024**3, "free_bytes": 8 * 1024**3}
    screen = scale.memory_preflight(64, inputs, output, memory, snapshot=enough)
    assert screen["actual_stacked_input_bytes"] == 4096
    assert screen["output_bytes_from_actual_B4_leaf_shapes"] == 1536
    assert screen["projected_device_buffer_bytes"] >= 1_600_000
    with pytest.raises(RuntimeError, match="slack"):
        scale.memory_preflight(64, inputs, output, memory, snapshot={**enough, "free_bytes": scale.GPU_SLACK_BYTES})


def test_scale_initial_state_allows_only_declared_bare_soil_sentinel(monkeypatch):
    lanes, second = prepared(16)
    for item in lanes:
        item.arguments.values_by_component[0]["rough"] = np.asarray([[np.nan, 2.0]], dtype=np.float64)
    contract = {f"[{block}][0][0]['rough']": {"inner_shape": [1, 2], "bare_soil_index": 0}
                for block in (0, 1)}
    monkeypatch.setattr(scale.base, "roughheight_sentinel_contract", lambda lane: contract)
    assert scale.screen_inputs(lanes, second)["unique_real_points"] == 16
    lanes[-1].arguments.values_by_component[0]["rough"][0, 1] = np.nan
    with pytest.raises(ValueError, match="initial-state nonfinite"):
        scale.screen_inputs(lanes, second)


def test_anchor_import_is_read_only_identity_checked_and_excludes_old_measurements(monkeypatch, tmp_path):
    source = tmp_path / "source"
    identity = {"science": "anchor evidence", "runtime": "same"}
    monkeypatch.setattr(scale.base, "stage_identity", lambda *args: identity)
    source_store = StageStore(source, identity)
    value = (np.ones((4, 2), dtype=np.float64), np.asarray([True] * 4))
    source_store.save("B4_batch_only_v1", {"outputs": value, "measurements": {"old_seconds": 1000}},
                      {"execution_origin": {"available": True, "window_identity": "old actual window"}})
    for point in scale.base.POINTS:
        source_store.save("scalar_" + point, np.ones(2, dtype=np.float64),
                          {"provenance": {"own_context_bound": True, "nested_cache_cleared_before_lower": True}})
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    args = SimpleNamespace(reference_dir=source)
    loaded, origins = scale.load_anchor_references(args, *prepared(16), {})
    assert isinstance(loaded["B4_batch_only_v1"], tuple)
    assert "old_seconds" not in repr(loaded)
    assert origins["references"]["B4_batch_only_v1"]["execution_origin"]["window_identity"] == "old actual window"
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    monkeypatch.setattr(scale.base, "stage_identity", lambda *args: {"science": "changed"})
    with pytest.raises(ValueError, match="identity mismatch"):
        scale.load_anchor_references(args, *prepared(16), {})
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before


@pytest.mark.parametrize("relation", ["same", "source_nested", "destination_nested"])
def test_scale_cli_rejects_reference_overlap_before_setup(monkeypatch, tmp_path, relation):
    source = tmp_path / "source"
    source.mkdir()
    marker = source / "report.json"
    marker.write_text("preserve original evidence")
    target = {"same": source, "source_nested": tmp_path, "destination_nested": source / "new"}[relation]
    monkeypatch.setattr("sys.argv", ["benchmark_global_scale.py", "--reference-dir", str(source),
                                    "--prepared-input", str(tmp_path / "input" / "prepared.pkl"),
                                    "--output-dir", str(target)])

    def forbidden(*args, **kwargs):
        pytest.fail("overlapping reference reached output/cache setup")

    monkeypatch.setattr(scale, "configure_cache", forbidden)
    monkeypatch.setattr(scale.base, "Recorder", forbidden)
    with pytest.raises(ValueError, match="non-overlapping"):
        scale.main()
    assert marker.read_text() == "preserve original evidence"
    assert list(source.iterdir()) == [marker]


def prepared_file(monkeypatch, tmp_path):
    config = tmp_path / "science.yaml"
    config.write_bytes(b"declared scientific configuration")
    run_def = tmp_path / "used_run.def"
    scale.base.write_materialized_run_def({"A": "1"}, run_def)
    lanes, second = prepared(16)
    lanes = [replace(item, context=SimpleNamespace(config_path=config, run_def_path=run_def,
                                                  run_def_values={"A": "1"})) for item in lanes]
    path = tmp_path / "input" / "prepared.pkl"
    path.parent.mkdir()
    path.write_bytes(pickle.dumps((lanes, second)))
    monkeypatch.setattr(batch, "static_signature", lambda *args: (("science", "rechecked"),))
    return path, config, lanes, second


def test_prepared_reader_hydrates_portable_paths_and_checks_declared_config(monkeypatch, tmp_path):
    path, config, lanes, second = prepared_file(monkeypatch, tmp_path)
    portable = PurePosixPath(config.as_posix())
    lanes = [replace(item, context=SimpleNamespace(config_path=portable,
                                                  run_def_path=item.context.run_def_path,
                                                  run_def_values=item.context.run_def_values)) for item in lanes]
    path.write_bytes(pickle.dumps((lanes, second)))
    read_lanes, read_second = scale.read_prepared(path, config)
    assert isinstance(read_lanes[0].context.config_path, Path)
    assert read_lanes[0].static_signature == (("science", "rechecked"),)
    assert scale.base.semantic_reference_identity(SimpleNamespace(baseline_ref="same"), read_lanes[0], read_second[0]) == (
        scale.base.semantic_reference_identity(SimpleNamespace(baseline_ref="same"),
                                               replace(lanes[0], static_signature=(("science", "rechecked"),)), second[0]))
    changed = tmp_path / "changed.yaml"
    changed.write_bytes(b"changed scientific configuration")
    with pytest.raises(ValueError, match="configuration differs"):
        scale.read_prepared(path, changed)
    monkeypatch.setattr(scale.os, "name", "posix")
    with pytest.raises(ValueError, match="unmapped Windows"):
        scale.PreparedUnpickler(io.BytesIO(pickle.dumps(PurePosixPath("E:/unmapped/input")))).load()


def test_scale_main_exercises_real_prepared_reader(monkeypatch, tmp_path):
    path, config, lanes, second = prepared_file(monkeypatch, tmp_path)
    reference = tmp_path / "reference"
    reference.mkdir()
    output = tmp_path / "result"
    monkeypatch.setattr("sys.argv", ["benchmark_global_scale.py", "--prepared-input", str(path),
                                    "--reference-dir", str(reference), "--output-dir", str(output),
                                    "--config", str(config), "--sizes", "16"])
    called = []

    def accepted(args, recorder, actual, others):
        called.append(actual)
        assert actual[0].static_signature == (("science", "rechecked"),)
        np.testing.assert_array_equal(others[0].science_day_numbers, second[0].science_day_numbers)

    monkeypatch.setattr(scale, "scale_benchmark", accepted)
    assert scale.main() == 0
    assert len(called) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "passed_bounded_scale_forward_diagnostic"
    assert any(item["phase"] == "prepared_input_reused" for item in report["phases"])


def local_cpu_evidence(monkeypatch, tmp_path):
    path, config, lanes, second = prepared_file(monkeypatch, tmp_path)
    lanes, second = scale.read_prepared(path, config)
    anchors, anchor_second = lanes[:4], second[:4]
    path.write_bytes(pickle.dumps((anchors, anchor_second)))
    reference = tmp_path / "reference"
    reference.mkdir()
    environment = {"python": "same", "jax": "same", "jaxlib": "same", "numpy": "same",
                   "device_details": [{"platform": "cpu", "kind": "cpu", "id": 0}],
                   "x64_enabled": True, "baseline_ref": "pinned", "git_head": "source",
                   "backend_platform_version": "current", "env": {key: None for key in (
                       "XLA_FLAGS", "JAX_PLATFORMS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    historical = {**environment, "backend_platform_version": None}
    current = {"status": "passed_bounded_forward_diagnostic", "environment": historical,
               "phases": [{"phase": "prepared_input_reused", "path": str(path)}],
               "gates": [{"name": name, "passed": True} for name in (
                   "B4_permutation", "B1_original_AOT_vs_extracted_full_state_day_outputs")]}
    origin = {"environment": historical, "phases": [
        {"phase": "independent_scalar_nested_cache_cleared", "landpoint_id": point, "entries_after": 0}
        for point in scale.base.POINTS]}
    (reference / "report.json").write_text(json.dumps(current))
    (reference / "reference_origin_report.json").write_text(json.dumps(origin))
    args = SimpleNamespace(reference_dir=reference, anchor_prepared_input=path, config=config, baseline_ref="pinned")
    for item, other in zip(anchors, anchor_second, strict=True):
        saved = scale.base.reference_identity(args, item, other)
        (reference / f"scalar_{item.landpoint_id}_reference.pkl").write_bytes(pickle.dumps((saved, np.ones(2))))
    saved = scale.base.reference_identity(args, anchors[0], anchor_second[0])
    output = np.ones((4, 2))
    (reference / "B4_candidate_before_gate.pkl").write_bytes(pickle.dumps((saved, output)))
    (reference / "B4_full_state_outputs.pkl").write_bytes(pickle.dumps(output))
    monkeypatch.setattr(scale.subprocess, "check_output", lambda *args, **kwargs: "")
    monkeypatch.setattr(scale.base, "roughheight_sentinel_contract", lambda item: {})
    return args, lanes, second, environment


def test_local_cpu_reference_migration_proves_inputs_and_discloses_missing_metadata(monkeypatch, tmp_path):
    args, lanes, second, environment = local_cpu_evidence(monkeypatch, tmp_path)
    before = {path.name: path.read_bytes() for path in args.reference_dir.iterdir()}
    values, evidence = scale.load_local_cpu_anchors(args, lanes, second, environment)
    assert len(values) == 5
    assert len(evidence["missing_origin_metadata"]) == 3
    assert "no historical stage manifest" in evidence["identity_hash_scope"]
    assert before == {path.name: path.read_bytes() for path in args.reference_dir.iterdir()}
    lanes[0].arguments.values_by_component[0]["x"][0] += 1
    with pytest.raises(ValueError, match="canonical anchor inputs changed"):
        scale.load_local_cpu_anchors(args, lanes, second, environment)


@pytest.mark.parametrize("change", ["gpu", "jax", "compiler"])
def test_local_cpu_reference_migration_rejects_runtime_changes(monkeypatch, tmp_path, change):
    args, lanes, second, environment = local_cpu_evidence(monkeypatch, tmp_path)
    if change == "gpu":
        environment["device_details"] = [{"platform": "gpu"}]
    elif change == "jax":
        environment["jax"] = "changed"
    else:
        environment["env"]["XLA_FLAGS"] = "changed"
    with pytest.raises(ValueError, match="restricted to CPU|runtime mismatch|compiler environment mismatch"):
        scale.load_local_cpu_anchors(args, lanes, second, environment)


@pytest.mark.parametrize("change", [None, "parameter", "config", "reference", "state", "vegetation"])
def test_remote_cold_state_reuse_proves_exact_source_and_only_declared_path_mapping(change):
    source = {"VEGETATION_FILE": "historical/source.nc", "science": "fixed"}
    normalized = {**source, "VEGETATION_FILE": "E:/local/source.nc"}
    state = SimpleNamespace(tstep=47, fields_by_component={"x": np.asarray([2.0], dtype=np.float64)})
    record = {"vegetation_file_path_only_mapping": {"local_value": normalized["VEGETATION_FILE"],
               "source_runtime_used_run_def_value": source["VEGETATION_FILE"]},
              "run_def_values_sha256": remote.values_digest(normalized), "reference_dir": "E:/local/reference",
              "cold_state_numeric_sha256": remote.producer._state_digest(state)}
    identity = (dict(normalized), b"exact science config", "E:/local/reference")
    record["cold_identity_sha256"] = remote.producer._cold_identity_hash(identity)
    record["config_sha256"] = hashlib.sha256(identity[1]).hexdigest()
    identity_hash = record["cold_identity_sha256"]
    config_bytes = identity[1]
    if change == "parameter":
        source["science"] = "changed"
    elif change == "config":
        config_bytes = b"changed"
    elif change == "reference":
        record["reference_dir"] = "another point reference"
    elif change == "state":
        state.fields_by_component["x"][0] += 1
    elif change == "vegetation":
        source["VEGETATION_FILE"] = "unauthorized/source.nc"
    if change is None:
        assert remote.validate_cold_source("point", state, source, record, config_bytes, identity_hash) is state
    else:
        with pytest.raises(ValueError, match="identity changed|unauthorized source"):
            remote.validate_cold_source("point", state, source, record, config_bytes, identity_hash)


def test_remote_rehydration_retains_anchors_and_admits_actual_full64_file(monkeypatch, tmp_path):
    path, config, anchors, anchor_second = prepared_file(monkeypatch, tmp_path)
    anchors, anchor_second = scale.read_prepared(path, config)
    anchors, anchor_second = anchors[:4], anchor_second[:4]
    path.write_bytes(pickle.dumps((anchors, anchor_second)))
    lanes, second = prepared(64)
    ids = [item.landpoint_id for item in lanes]
    source_vegetation = tmp_path / "source_vegetation.nc"
    source_vegetation.write_bytes(b"exact vegetation content")
    source_values = {point: {"VEGETATION_FILE": source_vegetation.as_posix(), "science": "fixed"} for point in ids}
    data_root = tmp_path / "data"
    vegetation = data_root / "INPUTDIR_ZZ/PFT1860_mangr_025deg.nc"
    vegetation.parent.mkdir(parents=True)
    vegetation.write_bytes(b"exact vegetation content")
    anchor_run_def = tmp_path / "anchor_used_run.def"
    scale.base.write_materialized_run_def({"VEGETATION_FILE": vegetation.as_posix()}, anchor_run_def)
    state_layout = SimpleNamespace(tstep=47, fields_by_component={"toy": {"x": np.asarray([1.0])}},
                                   provenance_by_component={"toy": {"x": "own source"}})
    restart = SimpleNamespace(stomate_readstart=SimpleNamespace(season_state=SimpleNamespace(provenance="own source")))
    anchors = [replace(item, initial_state=state_layout, context=SimpleNamespace(config_path=config,
        run_def_path=anchor_run_def, run_def_values={"VEGETATION_FILE": vegetation.as_posix()},
        first_step_restart_state=restart)) for item in anchors]
    path.write_bytes(pickle.dumps((anchors, anchor_second)))
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    assets = {point: {"runtime_run_def_sha256": "fixed", "reference_file_sha256": {"own": point}}
              for point in ids}
    cold, cold_identities, records = {}, {}, []
    for item, other in zip(lanes[4:], second[4:], strict=True):
        state = SimpleNamespace(tstep=47, fields_by_component={"toy": {"x": np.asarray([1.0])}},
                                provenance_by_component={"toy": {"x": "own source"}})
        values = {**source_values[item.landpoint_id], "VEGETATION_FILE": "E:/local/source.nc"}
        reference = "E:/local/reference/" + item.landpoint_id
        cold[item.landpoint_id] = state
        cold_identities[item.landpoint_id] = remote.producer._cold_identity_hash((values, config.read_bytes(), reference))
        records.append({"landpoint_id": item.landpoint_id, "reference_dir": reference,
            "cold_identity_sha256": cold_identities[item.landpoint_id],
            "config_sha256": remote.producer._sha256_file(config),
            "vegetation_file_path_only_mapping": {"local_value": "E:/local/source.nc",
                "source_runtime_used_run_def_value": source_vegetation.as_posix(),
                "local_file_sha256": remote.producer._sha256_file(vegetation)},
            "run_def_values_sha256": remote.values_digest(values),
            "cold_state_numeric_sha256": remote.producer._state_digest(state),
            "first_block_arguments_sha256": remote.producer._tree_digest(item.arguments),
            "second_block_arguments_sha256": remote.producer._tree_digest(other)})
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selections": {"64": ids}}))
    cold_path = tmp_path / "cold.pkl"
    cold_path.write_bytes(pickle.dumps(cold))
    cold_identity_path = tmp_path / "cold_identity.json"
    cold_identity_path.write_text(json.dumps(cold_identities))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "prepared", "selected_points": ids, "prepared_point_count": 64,
        "inventory_manifest_sha256": "inventory", "config_sha256": remote.producer._sha256_file(config),
        "asset_inputs": assets, "new_point_records": records,
        "cold_states_sha256": remote.producer._sha256_file(cold_path),
        "cold_identities_sha256": remote.producer._sha256_file(cold_identity_path)}))
    monkeypatch.setattr(remote.producer, "_read_inventory", lambda path: ({"_sha256": "inventory"}, source_values, {}))
    monkeypatch.setattr(remote.producer, "_verify_selected_assets", lambda *args: assets)
    monkeypatch.setattr(scale.base, "roughheight_sentinel_contract", lambda item: {})

    def context(config, used_run_def_path, reference_run_dir):
        return SimpleNamespace(config_path=config, run_def_path=used_run_def_path,
                               run_def_values=remote.producer.read_run_def_values(used_run_def_path),
                               first_step_restart_state=restart)

    def prepare_own(point, context, state, day_numbers=range(2, 9)):
        index = ids.index(point)
        arguments = lanes[index].arguments if list(day_numbers)[0] == 2 else second[index]
        return replace(lanes[index], context=context, initial_state=state, arguments=arguments,
                       static_signature=(("science", "rechecked"),))

    monkeypatch.setattr(remote.driver, "prepare_paper_1961_driver_context", context)
    monkeypatch.setattr(remote.batch, "prepare_lane", prepare_own)

    def forbidden(*args, **kwargs):
        pytest.fail("remote rehydration generated cold Day1 or compiled the transition")

    monkeypatch.setattr(remote.driver, "paper_1961_driver_cold_start_day_scaffold", forbidden)
    monkeypatch.setattr(remote.base, "lower_compile", forbidden)
    args = SimpleNamespace(inventory_tar=tmp_path / "inventory.tar.gz", selection=selection,
        preparation_audit=audit, cold_states=cold_path, cold_identities=cold_identity_path,
        anchor_prepared=path, asset_root=asset_root,
        data_root=data_root, config=config, output_dir=tmp_path / "rehydrated", max_seconds=30,
        baseline_ref="pinned")
    remote.rehydrate(args)
    report = json.loads((args.output_dir / "remote-preparation-audit.json").read_text())
    actual, others = scale.read_prepared(args.output_dir / "prepared_inputs_remote_64.pkl", config)
    assert report["status"] == "prepared" and len(actual) == 64
    assert len(report["new_point_records"]) == 60
    assert not report["cold_Day1_generated"] and not report["physical_transition_compiled"]
    assert report["actual_full64_reader_screen"]["unique_real_points"] == 64
    assert report["anchor_canonical_inputs"] == [scale.base.semantic_reference_identity(args, item, other)
        for item, other in zip(actual[:4], others[:4], strict=True)]
