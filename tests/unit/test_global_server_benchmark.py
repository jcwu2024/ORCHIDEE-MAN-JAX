"""Small scheduling/admission/cost tests, never a physical model forward run."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from jax_orchidee.driver import global_batch as batch
from scripts.dev import benchmark_global_scale as scale
from scripts.dev import benchmark_global_server_cpu as cpu
from scripts.dev import global_scale_candidate_first as candidate
from scripts.dev import global_server_resources as resources
from scripts.dev import summarize_global_server_cost as cost
from scripts.dev.global_batch_stage_store import StageStore
from scripts.dev.global_server_reference import load_reference


def topology(path, cores):
    for logical, core in enumerate(cores):
        target = path / f"cpu{logical}" / "topology"
        target.mkdir(parents=True)
        (target / "physical_package_id").write_text("0")
        (target / "core_id").write_text(str(core))
        (target / "thread_siblings_list").write_text(",".join(str(i) for i, other in enumerate(cores) if other == core))


def env(partition="cnall", count=4):
    return {"SLURM_JOB_ID": "123", "SLURM_JOB_PARTITION": partition, "SLURM_CPUS_PER_TASK": str(count),
            "SLURM_NTASKS": "1", "CUDA_VISIBLE_DEVICES": "1", "SLURM_JOB_GPUS": "1", "ORCHJAX_SLURM_ALLOC_TRES": "cpu=4,gres/gpu=1"}


def test_CPU_physical_count_and_GPU_auxiliary_SMT_are_different(tmp_path):
    topology(tmp_path, [0, 0, 1, 1])
    with pytest.raises(RuntimeError, match="unique physical"):
        resources.admit("pool", 4, environ=env(), allowed=[0, 1, 2, 3], sysfs=tmp_path, hostname="ibpaid")
    accepted = resources.admit_gpu(environ=env("gnall"), allowed=[0, 1, 2, 3], sysfs=tmp_path, hostname="ibpaid")
    assert accepted["allocated_CPU_request_count"] == 4
    assert accepted["observed_unique_physical_cores"] == 2
    assert accepted["selected_logical_cpus"] == [0, 1, 2, 3]
    assert accepted["CUDA_VISIBLE_DEVICES_preserved"] == "1"
    with pytest.raises(RuntimeError, match="real gnall"):
        resources.admit_gpu(environ=env("cnall"), allowed=[0], sysfs=tmp_path, hostname="ibpaid")
    with pytest.raises(RuntimeError, match="login/diagnostic"):
        resources.admit_gpu(environ=env("gnall"), allowed=[0], sysfs=tmp_path, hostname="ibtest02")


def test_thread_mask_census_rejects_backend_thread_widening(tmp_path):
    for tid, mask in [(1, "2"), (2, "2-3")]:
        path = tmp_path / str(tid)
        path.mkdir()
        (path / "status").write_text(f"Name: toy\nCpus_allowed_list:\t{mask}\n")
    with pytest.raises(RuntimeError, match="widened"):
        resources.thread_affinity([2], tmp_path)
    proof = resources.thread_affinity([2, 3], tmp_path)
    assert proof["observed_thread_count"] == 2
    assert resources.parse_cpu_list("0-2,5,8-9") == {0, 1, 2, 5, 8, 9}


def toy_prepared(size=64):
    ids = list(scale.base.POINTS) + [f"toy{index}" for index in range(4, size)]
    lanes, second = [], []
    for index, point in enumerate(ids):
        arguments = batch.BlockArguments(({"x": np.asarray([float(index)], dtype=np.float64),
                                            "mask": np.asarray([True]), "optional": None},),
            np.ones((7, 48, 1)), np.arange(2, 9, dtype=np.int32),
            *({"x": np.asarray(float(index))} for _ in range(6)))
        lanes.append(batch.PreparedLane(point, None, None, None, None, arguments, (("toy", 1),)))
        second.append(arguments._replace(science_day_numbers=np.arange(9, 16, dtype=np.int32)))
    return lanes, second


def toy_transition(*arguments):
    state = arguments[0][0]
    return ({**state, "x": state["x"] + arguments[3]["x"] + 1},), {"daily": arguments[1].sum(), "mask": state["mask"]}


def toy_environment(backend="cpu"):
    return {"jax": "0.4.38", "jaxlib": "0.4.38", "numpy": "2.2.6" if backend == "cpu" else "2.4.6",
            "x64_enabled": True, "device_details": [{"platform": backend}], "gpu_count": int(backend == "gpu")}


def toy_identity(lanes, backend="cpu"):
    return {"contract": "toy", "baseline_ref": "source", "science_sha256": "a" * 64,
            "precision_policy": scale.PRECISION_POLICY, "gate": {"atol": 1e-8, "rtol": 1e-10},
            "runtime": toy_environment(backend),
            "lanes": [{"landpoint_id": lane.landpoint_id, "inputs_sha256": str(index)} for index, lane in enumerate(lanes)]}


def mock_toy(monkeypatch, lanes, second):
    monkeypatch.setattr(scale, "screen_inputs", lambda *args: {"unique_real_points": len(lanes)})
    monkeypatch.setattr(scale.base, "legacy_factory", lambda *args: None)
    monkeypatch.setattr(scale.base, "roughheight_sentinel_contract", lambda *args: {})
    monkeypatch.setattr(batch, "make_scalar_transition", lambda *args, **kwargs: toy_transition)
    monkeypatch.setattr(scale.base, "stage_identity", lambda *args: toy_identity(lanes))
    monkeypatch.setattr(cpu, "thread_affinity", lambda *args: {"all_observed_thread_masks_within_declared_cores": True})
    monkeypatch.setattr(candidate, "thread_affinity", lambda *args: {"all_observed_thread_masks_within_declared_cores": True})
    outputs = [scale.base.run_two(jax.jit(toy_transition), lane.arguments, other)
               for lane, other in zip(lanes, second, strict=True)]
    combined = jax.tree_util.tree_map(lambda *parts: np.stack(parts), *outputs)
    return combined


def test_CPU_scalar_reuses_one_compile_across_all_unique16_and64(monkeypatch, tmp_path):
    lanes, second = toy_prepared()
    reference = mock_toy(monkeypatch, lanes, second)
    import scripts.dev.global_server_reference as refs

    monkeypatch.setattr(refs, "load_reference", lambda *args, **kwargs: (reference, {"sha256": "sealed"}))
    compiled = []
    def compile_toy(fn, arguments, name, recorder):
        compiled.append(name)
        return jax.jit(fn)
    monkeypatch.setattr(scale.base, "lower_compile", compile_toy)
    args = SimpleNamespace(mode="scalar", sizes=[16, 64], cpu_budget=1, warm_repeats=5,
        baseline_ref="source", gpu_reference_dir=tmp_path / "source", cpu_reference_dir=None, output_dir=tmp_path / "target")
    recorder = scale.base.Recorder(args.output_dir)
    recorder.result["environment"] = toy_environment()
    try:
        cpu.benchmark(args, recorder, lanes, second, scale, {"selected_logical_cpus": [0]})
    finally:
        recorder.stop.set()
    assert compiled == ["server_scalar_once"]
    manifest = json.loads((args.output_dir / "stage_manifest.json").read_text())
    assert [record["status"] for record in manifest["stages"].values()] == ["complete", "complete"]
    for size in (16, 64):
        value, record = StageStore(args.output_dir, manifest["identity"]).load(f"B{size}_server_scalar_v1")
        assert value["outputs"][0][0][0]["x"].shape == (size, 1)
        assert len(record["timing"][f"B{size}_warm_frozen_state_core"]["samples_seconds"]) == 5


def test_GPU_candidate_checkpoint_precedes_optional_B4_compile_and_survives_failure(monkeypatch, tmp_path):
    lanes, second = toy_prepared()
    reference = mock_toy(monkeypatch, lanes, second)
    monkeypatch.setattr(candidate, "load_reference", lambda *args, **kwargs: (reference, {"sha256": "sealed"}))
    monkeypatch.setattr(scale, "visible_gpu_memory", lambda: {"applicable": False})
    names = []
    def compile_toy(fn, arguments, name, recorder):
        names.append(name)
        if name == "B4_resident_after_candidates":
            manifest = json.loads((tmp_path / "target/stage_manifest.json").read_text())
            assert manifest["stages"]["B64_candidate_first_v1"]["status"] == "complete"
            assert manifest["stages"]["B16_candidate_first_v1"]["status"] == "complete"
            raise InterruptedError("bounded optional baseline stop")
        return jax.jit(fn)
    monkeypatch.setattr(scale.base, "lower_compile", compile_toy)
    args = SimpleNamespace(sizes=[64, 16], warm_repeats=5, output_dir=tmp_path / "target", full_reference_dir=tmp_path / "source",
                           baseline_ref="source", measure_b4_baseline=True)
    recorder = scale.base.Recorder(args.output_dir)
    recorder.result.update(environment=toy_environment("gpu"), server_resources={"original_affinity": [0]})
    try:
        with pytest.raises(InterruptedError, match="optional"):
            candidate.candidate_first_benchmark(args, recorder, lanes, second)
    finally:
        recorder.stop.set()
    assert names == ["B64_candidate", "B16_candidate", "B4_resident_after_candidates"]
    manifest = json.loads((args.output_dir / "stage_manifest.json").read_text())
    assert manifest["stages"]["optional_new_B4_comparison_v1"]["status"] == "stopped"
    assert recorder.result["all_required_stages_complete"] is True


def sealed_reference(tmp_path):
    lanes, _ = toy_prepared(16)
    identity = toy_identity(lanes, "gpu")
    store = StageStore(tmp_path, identity)
    store.save("B4_group_references", np.arange(16), {"provenance": {
        "point_ids": [lane.landpoint_id for lane in lanes], "own_inputs": True}, "gates": [{"name": "valid", "passed": True}]})
    return identity


def test_portable_reference_retains_backend_conditions_and_rejects_identity_or_checksum(tmp_path):
    identity = sealed_reference(tmp_path)
    current = copy.deepcopy(identity)
    current["runtime"] = toy_environment()
    value, origin = load_reference(tmp_path, current)
    assert value.tolist() == list(range(16))
    assert origin["backend_conditions"]["source"]["numpy"] == "2.4.6"
    assert origin["backend_conditions"]["current"]["numpy"] == "2.2.6"
    bad = copy.deepcopy(current)
    bad["lanes"][:2] = bad["lanes"][:2][::-1]
    with pytest.raises(ValueError, match="point order"):
        load_reference(tmp_path, bad)
    with (tmp_path / "stage_B4_group_references.pkl").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_reference(tmp_path, current)


def test_cost_terminal_TIMEOUT_is_aggregate_and_K_model_adds_saved_IO():
    stage = {"directory": "/accepted", "stage": "B16_server_scalar_v1", "CPU_count": 1, "GPU_count": 0,
             "core": {"median_seconds": 2, "point_days": 224, "samples_seconds": [2] * 5},
             "wrapper": {"median_seconds": 3, "point_days": 224, "samples_seconds": [3] * 5},
             "saved_output_checkpoint_IO": {"available": True, "serialization_seconds": 0.2, "atomic_write_seconds": 0.3}}
    job = {"directories": ["/accepted", "/other-mode"], "status": "TIMEOUT", "wall_seconds": 3600,
           "allocated_CPU_count": 1, "GPU_count": 0, "provenance": "Slurm accounting", "reason": "1h ceiling",
           "stage_startup_and_validation_seconds": {stage["stage"]: 10}}
    result = cost.scenarios(stage, 0.05, 2.2, [1, 10], job)
    aggregate = result["aggregate_benchmark_job_cost"]
    assert aggregate["CNY"] == 0.05 and aggregate["observed_terminal_status"] == "TIMEOUT"
    assert "CNY_per_common_workload" not in aggregate
    assert result["assumed_K_reuse"][1]["wall_seconds"] == 35
    assert result["assumed_K_reuse"][1]["assumed_K_is_observed"] is False
    job["status"] = "RUNNING"
    with pytest.raises(ValueError, match="accounting mismatch"):
        cost.scenarios(stage, 0.05, 2.2, [1], job)


@pytest.mark.parametrize("invalid", ["samples", "denominator", "nonfinite"])
def test_cost_rejects_incomplete_or_wrong_workload_metrics(invalid):
    value = {"samples_seconds": [2] * 5, "median_seconds": 2, "valid_point_days": 224}
    if invalid == "samples":
        value["samples_seconds"] = [2] * 3
    elif invalid == "denominator":
        value["valid_point_days"] = 896
    else:
        value["samples_seconds"][0] = float("nan")
    with pytest.raises(ValueError):
        cost.metrics(value, 224)


def test_cost_allocation_uses_CPU_TRES_instead_of_assuming_physical_equals_billable():
    count, proof = cost.allocated_cpu_count({"allocated_CPU_request_count": 4,
        "Slurm": {"ORCHJAX_SLURM_ALLOC_TRES": "cpu=8,gres/gpu=1"}})
    assert count == 8 and proof["requested_CPU_count"] == 4


def test_pool_core_barrier_contains_all_workers_without_full_output_IPC():
    sent = []
    class Channel:
        def __init__(self, indices):
            self.indices = indices
        def send(self, value):
            sent.append(value)
        def recv(self):
            return {"completed_point_indices": self.indices}
    runner = cpu.PoolRunner.__new__(cpu.PoolRunner)
    runner.channels = [Channel(list(range(index, 16, 4))) for index in range(4)]
    runner.run(16)
    assert sent == [{"command": "run", "size": 16}] * 4
    runner.channels[-1].indices = []
    with pytest.raises(ValueError, match="every unique point"):
        runner.run(16)


def test_pool_wrapper_transfers_fresh_host_pairs_and_materializes_full_outputs(monkeypatch, tmp_path):
    lanes, second = toy_prepared(16)
    import scripts.dev.probe_global_batch_cache as cache

    monkeypatch.setattr(cpu, "bind", lambda *args, **kwargs: None)
    monkeypatch.setattr(cpu, "thread_affinity", lambda *args: {"within_mask": True})
    monkeypatch.setattr(scale, "read_prepared", lambda *args: (lanes, second))
    monkeypatch.setattr(scale, "screen_inputs", lambda *args: {})
    monkeypatch.setattr(scale.base, "metadata", lambda *args: toy_environment())
    monkeypatch.setattr(scale.base, "independent_scalar_executable", lambda *args: jax.jit(toy_transition))
    monkeypatch.setattr(cache, "configure_cache", lambda *args: {})
    monkeypatch.setattr(cache, "CacheEvents", lambda: SimpleNamespace(close=lambda: None))
    puts, gets = [], []
    original_put, original_get = jax.device_put, jax.device_get
    def put(value):
        puts.append(value)
        return original_put(value)
    def get(value):
        gets.append(value)
        return original_get(value)
    monkeypatch.setattr(jax, "device_put", put)
    monkeypatch.setattr(jax, "device_get", get)
    class Channel:
        commands = iter([{"command": "run", "size": 16}, {"command": "wrapper", "size": 16}, {"command": "close"}])
        def __init__(self):
            self.sent = []
        def recv(self):
            return next(self.commands)
        def send(self, value):
            self.sent.append(value)
        def close(self):
            pass
    channel = Channel()
    args = SimpleNamespace(synthetic_smoke=False, cache_mode="disabled", cache_root=None,
        output_dir=tmp_path, prepared_input=tmp_path / "toy", config=tmp_path / "toy.yaml")
    cpu.pool_worker(channel, 0, args, [0, 4, 8, 12])
    assert not any("error" in item for item in channel.sent)
    assert len(puts) == 2 and puts[0] is not puts[1]
    assert puts[1][4][0] is lanes[4].arguments
    assert channel.sent[1] == {"completed_point_indices": [0, 4, 8, 12]}
    assert len(gets) == 1 and set(channel.sent[2]["outputs"]) == {0, 4, 8, 12}


def cost_fixture(tmp_path, backend="cpu"):
    points = [f"point{index}" for index in range(16)]
    identity = {"contract": "server-candidate-first-full-physics-v1" if backend == "gpu" else "server-CPU-same-workload-v1",
        "mode": "scalar", "lanes": [{"landpoint_id": point} for point in points],
        "precision_policy": cost.PRECISION, "science_sha256": "a" * 64, "baseline_ref": "source", "gate": cost.FIXED_GATE.copy()}
    name = "B16_candidate_first_v1" if backend == "gpu" else "B16_server_scalar_v1"
    gates = [f"B16_lane_{index}_{point}_both_blocks" if backend == "gpu" else f"B16_scalar_lane_{index}_{point}_GPU_reference"
             for index, point in enumerate(points)]
    gates += ["B16_cross_group_permutation", "B16_replacement_all_lanes_isolation"] if backend == "gpu" else ["B16_scalar_frozen_reset_repeat"]
    path = tmp_path / "complete.pkl"
    path.write_bytes(b"synthetic sealed snapshot, not a model")
    record = {"status": "complete", "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "gates": [{"name": name, "passed": True} for name in gates],
        "provenance": {"unique_real_points": 16, "point_ids": points, "precision_policy": cost.PRECISION,
            "blocks": [[2, 8], [9, 15]], "half_hours_per_day": 48, "full_outputs": True,
            "full_two_block_parity": True, "five_or_more_synchronized_repeats": True,
            "frozen_reset_repeat": True, "cross_mode_scalar_parity": True,
            "cross_group_permutation": True, "replacement_isolation": True},
        "execution_origin": {"environment": toy_environment(backend)},
        "timing": {key: {"samples_seconds": [2] * 5, "median_seconds": 2, "valid_point_days": 224}
            for key in ("B16_warm_frozen_state_core", "B16_preprepared_wrapper_stack_transfer_materialize")}}
    manifest = {"identity": identity, "stages": {name: record}}
    report = {"server_resources": {"allocated_CPU_request_count": 4, "Slurm": env("gnall" if backend == "gpu" else "cnall")}}
    report["server_resources"]["Slurm"]["ORCHJAX_SLURM_ALLOC_TRES"] = f"cpu=4,gres/gpu={int(backend == 'gpu')}"
    return name, manifest, report


def test_cost_complete_SHA_gates_and_actual_GPU_allocation_are_required(tmp_path):
    name, manifest, report = cost_fixture(tmp_path, "gpu")
    assert cost.accept_stage(tmp_path, name, manifest, report)["GPU_count"] == 1
    manifest["identity"]["gate"]["rtol"] = 1e-5
    with pytest.raises(ValueError, match="fixed tolerance"):
        cost.accept_stage(tmp_path, name, manifest, report)
    manifest["identity"]["gate"]["rtol"] = 1e-10
    report["server_resources"]["Slurm"]["ORCHJAX_SLURM_ALLOC_TRES"] = "cpu=4,gres/gpu=2"
    with pytest.raises(ValueError, match="GPU allocation"):
        cost.accept_stage(tmp_path, name, manifest, report)
    report["server_resources"]["Slurm"]["ORCHJAX_SLURM_ALLOC_TRES"] = "cpu=4,gres/gpu=1"
    record = manifest["stages"][name]
    record["gates"].pop()
    with pytest.raises(ValueError, match="gate set"):
        cost.accept_stage(tmp_path, name, manifest, report)
    record["status"] = "running"
    with pytest.raises(ValueError, match="incomplete"):
        cost.accept_stage(tmp_path, name, manifest, report)
    record["status"] = "complete"
    (tmp_path / record["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA"):
        cost.accept_stage(tmp_path, name, manifest, report)


def test_cost_invalid_job_accounting_preserves_valid_warm_stage(tmp_path):
    name, manifest, report = cost_fixture(tmp_path)
    (tmp_path / "stage_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "report.json").write_text(json.dumps(report))
    result = cost.summarize([tmp_path], [{"directories": [str(tmp_path.resolve())], "status": "RUNNING"}])
    assert len(result["accepted_stages"]) == 1 and not result["withheld_stages"]
    scenario = result["accepted_stages"][0]["cost_scenarios"]["primary"]
    assert scenario["core"]["CNY"] > 0 and scenario["aggregate_benchmark_job_cost"] is None
    assert "accounting mismatch" in scenario["job_accounting_withheld_reason"]
