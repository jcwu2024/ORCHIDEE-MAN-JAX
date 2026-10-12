"""Capacity protocol tests with tiny NumPy outputs; no physics or JIT compile."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from scripts.dev import benchmark_global_batch as actual_base
from scripts.dev import benchmark_global_gpu_capacity as capacity
from scripts.dev.global_batch_stage_store import StageStore


def toy_case(monkeypatch):
    lanes = [SimpleNamespace(landpoint_id=f"point{i}", arguments=np.array([[i]], dtype=np.float64)) for i in range(64)]
    other = [lane.arguments + 1 for lane in lanes]
    expanded, seconds, origins = capacity.capacity_layout(lanes, other)
    events = []
    recorder = SimpleNamespace(result={"gates": [], "timing": {}, "server_resources": {}, "environment": {}},
        sentinel_paths={}, phase=lambda *a, **kw: None)
    def lower(fn, arguments, name, recorder):
        events.append(("compile", len(arguments)))
        recorder.result["timing"].update({name + "_lower_seconds": 1, name + "_compile_seconds": 2})
        return fn
    def run(fn, first, second):
        events.append(("run", len(first)))
        return first.copy(), second.copy()
    def stack(selected):
        return np.concatenate([lane.arguments for lane in selected])
    fake_base = SimpleNamespace(batch=SimpleNamespace(stack_lanes=stack, make_batch_transition=lambda lanes: None),
        stack_second=lambda lanes, second: np.concatenate(second), lower_compile=lower, run_two=run,
        sync=lambda value: events.append(("sync", None)), compare=actual_base.compare,
        driver=SimpleNamespace(_COMPILED_SECHIBA_SCAN_CACHE={}), linux_ru_maxrss_bytes=lambda: 100, rss_bytes=lambda: 100)
    fake_jax = SimpleNamespace(device_put=lambda value: value, device_get=lambda value: value, tree_util=jax.tree_util)
    scale = SimpleNamespace(base=fake_base, jax=fake_jax, np=np,
        tree_bytes=lambda value: sum(np.asarray(leaf).nbytes for leaf in jax.tree_util.tree_leaves(value)),
        executable_memory=lambda value: {"available": True, "temp_size_in_bytes": 100},
        visible_gpu_memory=lambda: {"applicable": True, "total_bytes": 32 * 1024**3, "free_bytes": 30 * 1024**3})
    monkeypatch.setattr(capacity, "host_screen", lambda *args: {"passed": True})
    reference = np.arange(64, dtype=np.float64).reshape(64, 1)
    return expanded, seconds, (reference, reference + 1), origins, recorder, scale, events


@pytest.mark.parametrize("size,chunks", [(128, 4), (256, 2), (512, 1)])
def test_each_shape_measures_exact_same512_and_preserves_every_output(monkeypatch, tmp_path, size, chunks):
    lanes, second, reference, origins, recorder, scale, events = toy_case(monkeypatch)
    value, record = capacity.measure_shape(size, SimpleNamespace(warm_repeats=5), recorder, lanes, second,
        reference, scale, {64: {"available": True, "temp_size_in_bytes": 100, "old_benchmark_peak_RSS_bytes": 100}})
    assert events[0] == ("sync", None) and events[1] == ("compile", size)
    assert sum(event[0] == "compile" for event in events) == 1
    assert sum(event[0] == "run" for event in events) == 11 * chunks + 2
    assert value["outputs"][0].shape == (512, 1)
    assert np.array_equal(value["outputs"][0][:, 0], np.tile(np.arange(64), 8))
    assert record["provenance"]["chunks_per_collection"] == chunks
    for metric in value["measurements"].values():
        assert len(metric["samples_seconds"]) == 5 and metric["lane_days"] == 7168
        assert metric["unique_point_days"] == 896 and not metric["unique_point_days_is_throughput_denominator"]
    assert len(record["gates"]) == 515 and record["scientific_acceptance"] is False
    store = StageStore(tmp_path, {"contract": capacity.CONTRACT, "lane_origins": origins})
    store.save("completed", value, record)
    restored, proof = store.load("completed")
    assert restored["outputs"][0].shape[0] == 512
    assert proof["measurement_status"] == "complete" and proof["checkpoint_io"]["available"]


def test_finite_reference_tolerance_preserves_exploratory_samples(monkeypatch):
    lanes, second, reference, _, recorder, scale, _ = toy_case(monkeypatch)
    reference[0][0, 0] += 1e-5
    value, record = capacity.measure_shape(128, SimpleNamespace(warm_repeats=5), recorder, lanes, second,
        reference, scale, {64: {"available": True, "temp_size_in_bytes": 100, "old_benchmark_peak_RSS_bytes": 100}})
    failed = [gate for gate in record["gates"] if not gate["passed"]]
    assert len(failed) == 8 and all(gate["exploratory_continuation"] for gate in failed)
    assert record["strict_numerical_passed"] is False and len(value["measurements"]["core"]["samples_seconds"]) == 5


def test_hard_failure_is_recorded_for_all_lanes_and_keeps_previous_checkpoint(monkeypatch, tmp_path):
    lanes, second, reference, _, recorder, scale, events = toy_case(monkeypatch)
    reference[0][0, 0] = np.nan
    store = StageStore(tmp_path, {"contract": capacity.CONTRACT})
    store.save("prior128", {"outputs": "preserved"}, {"measurement_status": "complete"})
    with pytest.raises(AssertionError, match="hard per-lane"):
        store.run("later256", lambda: capacity.measure_shape(256, SimpleNamespace(warm_repeats=5),
            recorder, lanes, second, reference, scale,
            {64: {"available": True, "temp_size_in_bytes": 100, "old_benchmark_peak_RSS_bytes": 100}}))
    assert len(recorder.result["gates"]) == 512
    assert sum(event[0] == "run" for event in events) == 2
    assert store.load("prior128")[0]["outputs"] == "preserved"
    assert store.manifest["stages"]["later256"]["status"] == "failed"


@pytest.mark.parametrize("boundary", ["core", "wrapper"])
def test_conditional512_requires_both_boundaries_memory_and_time(boundary):
    records = {128: {"core": {"median_seconds": 10}, "wrapper": {"median_seconds": 10}},
               256: {"core": {"median_seconds": 9}, "wrapper": {"median_seconds": 9}}}
    assert capacity.conditional_512_decision(records, 1800, 100, 3 * 1024**3)["permitted"]
    records[256][boundary]["median_seconds"] = 9.6
    assert not capacity.conditional_512_decision(records, 1800, 100, 3 * 1024**3)["permitted"]
    assert not capacity.conditional_512_decision(records, 1799, 100, 3 * 1024**3)["permitted"]
    assert not capacity.conditional_512_decision(records, 1800, None, 3 * 1024**3)["permitted"]


def test_capacity_identity_rejects_duplicate_source_points_and_bad_sample_count():
    lanes = [SimpleNamespace(landpoint_id="same") for _ in range(64)]
    with pytest.raises(ValueError, match="unique FULL64"):
        capacity.capacity_layout(lanes, [None] * 64)
    with pytest.raises(ValueError, match="five positive"):
        capacity.lane_metrics([1] * 4)


def test_real_Bash_and_Python_capacity_dryrun_without_GPU_initialization(tmp_path):
    root = Path(capacity.__file__).resolve().parents[2]
    env = os.environ.copy()
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES", "CONDITIONAL_512", "OPTIONAL_B64_BASELINE"):
        env.pop(key, None)
    launcher = root / "scripts/hpc/slurm_global_gpu_capacity.sh"
    result = subprocess.run([shutil.which("bash"), str(launcher).replace("\\", "/"), "--dry-run"],
        env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    row = next(line for line in result.stdout.splitlines() if line.startswith("command="))
    tokens = shlex.split(row.removeprefix("command="))
    entry = next(i for i, token in enumerate(tokens) if token.endswith("global_batch_gpu_entry.py"))
    dry = subprocess.run([sys.executable, str(Path(capacity.__file__).resolve()), *tokens[entry + 1:], "--dry-run"],
        cwd=tmp_path, text=True, capture_output=True, check=False)
    assert dry.returncode == 0, dry.stderr
    report = json.loads(dry.stdout)
    assert report["candidates"] == [128, 256] and report["conditional_512"] is True
    assert report["total_lanes"] == 512 and report["unique_real_points"] == 64
    assert report["JAX_imported"] is False and report["model_execution"] is False


@pytest.mark.parametrize("report,expected", [
    ({"peak_rss_bytes": 12855533568, "linux_ru_maxrss_peak_bytes": 12869337088}, 12869337088),
    ({"peak_rss_bytes": 12855533568}, 12855533568),
])
def test_old_B64_memory_admission_uses_actual_report_field_names(tmp_path, report, expected):
    manifest = {"identity": {}, "stages": {"B64_candidate_first_v1": {
        "status": "complete", "provenance": {"executable_memory_analysis": {
            "available": True, "temp_size_in_bytes": 291230416}}}}}
    (tmp_path / "stage_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "report.json").write_text(json.dumps(report))
    _, _, analysis = capacity.baseline_memory(tmp_path)
    assert analysis["old_benchmark_peak_RSS_bytes"] == expected
    assert capacity.memory_estimate(128, 100, 200, {64: analysis}) == 2 * 100 + 3 * 200 + 2 * 291230416


@pytest.mark.parametrize("fail256", [False, True])
def test_benchmark_seals_each_candidate_before_next_then_conditional_and_baseline(monkeypatch, tmp_path, fail256):
    _, _, reference, _, recorder, scale, _ = toy_case(monkeypatch)
    lanes = [SimpleNamespace(landpoint_id=f"point{i}", arguments=np.array([[i]], dtype=np.float64)) for i in range(64)]
    second = [lane.arguments + 1 for lane in lanes]
    args = SimpleNamespace(output_dir=tmp_path / "new", full_reference_dir=tmp_path / "reference",
        memory_baseline_dir=tmp_path / "memory", prepared_input=tmp_path / "input" / "prepared.pkl",
        conditional_512=True, optional_b64_baseline=True, budget_seconds=5400, baseline_ref="source")
    analysis = {"available": True, "temp_size_in_bytes": 100, "old_benchmark_peak_RSS_bytes": 100}
    monkeypatch.setattr(capacity, "admit_numerical_sources", lambda *args: ({}, reference, {"sha256": "sealed"}, {"sha256": "old"}, analysis))
    scale.PRECISION_POLICY = "native"
    scale.screen_inputs = lambda *args: {}
    scale.base.legacy_factory = lambda *args: None
    scale.base.roughheight_sentinel_contract = lambda *args: {}
    scale.jax.clear_caches = lambda: None
    recorder.phase = lambda *args, **kwargs: None
    recorder.flush = lambda: None
    order = []
    def measure(size, args, recorder, lanes, other, reference, scale, memories):
        for prior in order:
            assert (args.output_dir / f"stage_capacity_B{prior}_fixed512_v1.pkl").is_file()
        if size == 256 and fail256:
            raise RuntimeError("later candidate failed")
        order.append(size)
        memories[size] = analysis
        seconds = {128:10, 256:9, 512:8, 64:15}[size]
        value = {"outputs": np.arange(512), "measurements": {"core": capacity.lane_metrics([seconds]*5), "wrapper": capacity.lane_metrics([seconds]*5)}}
        return value, {"measurement_status": "complete", "scientific_acceptance": False}
    monkeypatch.setattr(capacity, "measure_shape", measure)
    if fail256:
        with pytest.raises(RuntimeError, match="later candidate"):
            capacity.benchmark(args, recorder, lanes, second, scale)
        assert order == [128]
        manifest = json.loads((args.output_dir / "stage_manifest.json").read_text())
        assert manifest["stages"]["capacity_B128_fixed512_v1"]["status"] == "complete"
        assert manifest["stages"]["capacity_B256_fixed512_v1"]["status"] == "failed"
    else:
        capacity.benchmark(args, recorder, lanes, second, scale)
        assert order == [128, 256, 512, 64]
        assert recorder.result["conditional_512_decision"]["permitted"]


def test_continuation512_ignores_speedup_but_keeps_resource_guards():
    records = {128: {"core": {"median_seconds": 10}, "wrapper": {"median_seconds": 10}},
               256: {"core": {"median_seconds": 15}, "wrapper": {"median_seconds": 15}}}
    assert not capacity.conditional_512_decision(records, 1800, 100, 3*1024**3)["permitted"]
    assert capacity.conditional_512_decision(records, 1800, 100, 3*1024**3, continuation=True)["permitted"]
    for seconds, estimate, free in [(1799, 100, 3*1024**3), (1800, None, 3*1024**3), (1800, 100, 2*1024**3)]:
        assert not capacity.conditional_512_decision(records, seconds, estimate, free, continuation=True)["permitted"]


def test_continuation_measures_only256_then512_and_seals_first(monkeypatch, tmp_path):
    _, _, reference, _, recorder, scale, _ = toy_case(monkeypatch)
    lanes = [SimpleNamespace(landpoint_id=f"point{i}", arguments=np.array([[i]], dtype=np.float64)) for i in range(64)]
    second = [lane.arguments+1 for lane in lanes]
    args = SimpleNamespace(output_dir=tmp_path/'new', full_reference_dir=tmp_path/'reference',
        memory_baseline_dir=tmp_path/'memory', prepared_input=tmp_path/'input'/'prepared.pkl',
        b128_baseline_dir=tmp_path/'old128', continue_256_512=True, conditional_512=False,
        optional_b64_baseline=False, budget_seconds=5400, baseline_ref='source')
    analysis = {"available":True,"temp_size_in_bytes":100,"old_benchmark_peak_RSS_bytes":100}
    monkeypatch.setattr(capacity, 'admit_numerical_sources', lambda *a: ({},reference,{"sha256":"sealed"},{"sha256":"old"},analysis))
    baseline = {k:capacity.lane_metrics([1]*5) for k in ('core','wrapper')}
    monkeypatch.setattr(capacity,'import_b128_baseline',lambda *a, **kw: (baseline,{"role":"historical"}))
    scale.PRECISION_POLICY='native'
    scale.screen_inputs=lambda *a: {}
    scale.base.legacy_factory=lambda *a: None
    scale.base.roughheight_sentinel_contract=lambda *a: {}
    scale.jax.clear_caches=lambda: None
    recorder.phase=lambda *a,**k: None
    recorder.flush=lambda: None
    order=[]
    def measure(size,args,recorder,*other):
        if size==512:
            assert (args.output_dir/'stage_capacity_B256_fixed512_v1.pkl').is_file()
        order.append(size)
        other[-1][size]=analysis
        measurements={k:capacity.lane_metrics([2]*5) for k in ('core','wrapper')}
        return {"measurements":measurements},{"measurement_status":"complete"}
    monkeypatch.setattr(capacity,'measure_shape',measure)
    capacity.benchmark(args,recorder,lanes,second,scale)
    assert order==[256,512] and recorder.result['completed_shapes']==[256,512]
    manifest=json.loads((args.output_dir/'stage_manifest.json').read_text())
    assert manifest['identity']['contract']==capacity.CONTINUATION_CONTRACT
    assert 'capacity_B128_fixed512_v1' not in manifest['stages']
    assert recorder.result['comparison_shapes_including_imported_baseline']==[128,256,512]


@pytest.mark.parametrize("copied_to_slurm_spool", [False, True])
def test_continuation_launcher_actual_Bash_expansion(tmp_path, copied_to_slurm_spool):
    root=Path(capacity.__file__).resolve().parents[2]
    bash=shutil.which('bash')
    resolved=subprocess.run([bash,'-c','cd "$1" && pwd','path',str(tmp_path).replace('\\','/')],
        text=True,capture_output=True,check=True).stdout.strip()
    declared=tmp_path/'runtime/worktrees/declared-code/scripts/hpc'
    declared.mkdir(parents=True)
    project='/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX'
    for name in ('slurm_global_gpu_capacity.sh','slurm_global_gpu_capacity_continue.sh'):
        source=(root/'scripts/hpc'/name).read_text().replace(f'PROJECT={project}',f'PROJECT={shlex.quote(resolved)}')
        (declared/name).write_text(source)
    launcher=declared/'slurm_global_gpu_capacity_continue.sh'
    if copied_to_slurm_spool:
        spool=tmp_path/'slurm-spool/job15643327'
        spool.mkdir(parents=True)
        launcher=spool/'slurm_script'
        shutil.copyfile(declared/'slurm_global_gpu_capacity_continue.sh',launcher)
        assert not (spool/'slurm_global_gpu_capacity.sh').exists()
    env=os.environ.copy()
    for key in ('SLURM_CPUS_PER_TASK','SLURM_JOB_GPUS','CUDA_VISIBLE_DEVICES','CONTINUE_256_512','OPTIONAL_B64_BASELINE'):
        env.pop(key,None)
    env['CODE_DIR']=resolved+'/runtime/worktrees/declared-code'
    env['RUN_ROOT']=resolved+'/runtime/outputs/research/global_throughput/server-benchmark/new-only'
    result=subprocess.run([bash,str(launcher).replace('\\','/'),'--dry-run'],env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    tokens=shlex.split(next(line for line in result.stdout.splitlines() if line.startswith('command=')).removeprefix('command='))
    entry=next(i for i,t in enumerate(tokens) if t.endswith('global_batch_gpu_entry.py'))
    assert tokens[entry]==env['CODE_DIR']+'/scripts/hpc/global_batch_gpu_entry.py'
    dry=subprocess.run([sys.executable,str(Path(capacity.__file__).resolve()),*tokens[entry+1:],'--dry-run'],cwd=tmp_path,text=True,capture_output=True)
    assert dry.returncode==0,dry.stderr
    report=json.loads(dry.stdout)
    assert report['candidates']==[256,512] and report['B512_requires_speedup'] is False
    assert report['optional_B64'] is False and report['JAX_imported'] is False
    assert not (tmp_path/'runtime/outputs').exists()


@pytest.mark.parametrize("size", [2, 3])
def test_actual_JAX_two_block_bound_AOT_and_early_signature_audit(size):
    from jax_orchidee.driver.global_batch import BlockArguments
    counts=(350,14,1,4,10,20,60,38,39)
    groups=tuple(tuple(np.full((size,1),i+j*.01,dtype=np.float64) for j in range(c)) for i,c in enumerate(counts))
    first=BlockArguments(*groups)
    second=first._replace(forcing_days=tuple(v+1 for v in first.forcing_days))
    def transition(*arguments):
        forcing=arguments[1][0]
        return tuple(v+forcing for v in arguments[0]), {"all_state":arguments[0]}
    phases=[]
    recorder=SimpleNamespace(result={"timing":{}},phase=lambda name:phases.append(name))
    executable=actual_base.lower_compile(transition,first,'tiny',recorder)
    a,b=actual_base.run_two(executable,first,second)
    expected_a=tuple(v+first.forcing_days[0] for v in first.values_by_component)
    expected_b=tuple(v+second.forcing_days[0] for v in expected_a)
    for got,want in zip(a[0],expected_a,strict=True):
        np.testing.assert_array_equal(got,want)
    for got,want in zip(b[0],expected_b,strict=True):
        np.testing.assert_array_equal(got,want)
    metadata=recorder.result['compilations']['tiny']
    assert metadata['lowered_input_signature']['caller_leaves']==536
    assert metadata['compiled_input_signature']['internal_signature_equal'] is not False
    # Protect ownstage binding against an unrelated lazylayer call. This is a
    # fault-injection regression, not a natural reproduction of job15600405.
    def wrong(*args): raise TypeError('Computation compiled for 673 inputs but called with 536')
    executable.compiled._call=wrong
    repeated=actual_base.run_two(executable,first,second)
    for actual,expected in zip(jax.tree_util.tree_leaves(repeated),jax.tree_util.tree_leaves((a,b)),strict=True):
        np.testing.assert_array_equal(actual,expected)
    lowered=jax.jit(transition).lower(*first)
    with pytest.raises(TypeError,match='public input'):
        actual_base.audit_aot_signature(lowered,first._replace(science_day_numbers=(np.ones((size+1,1)),)),lowered=True)
    assert callable(executable.memory_analysis)


def test_internal_signature_mismatch_stops_before_expensive_compile(monkeypatch):
    arguments=(np.ones((2,1),dtype=np.float64),)
    tree,_=actual_base._argument_signature(arguments)
    aval=jax.ShapeDtypeStruct((2,1),np.float64)
    compile_called=[]
    lowered=SimpleNamespace(in_tree=tree,in_avals=((aval,),{}),
        _lowering=SimpleNamespace(compile_args={"all_args_info":SimpleNamespace(in_avals=[aval,aval]),"kept_var_idx":{0}}),
        compile=lambda:compile_called.append(True))
    monkeypatch.setattr(actual_base.jax,'jit',lambda fn:SimpleNamespace(lower=lambda *a:lowered))
    recorder=SimpleNamespace(result={"timing":{}},phase=lambda name:None)
    with pytest.raises(TypeError,match='internal input signature'):
        actual_base.lower_compile(lambda x:x,arguments,'mismatch',recorder)
    assert compile_called==[]
    metadata = recorder.result["compilations"]["mismatch"]
    assert metadata["lowered_input_metadata"]["caller_leaves"] == 1
    assert metadata["lowered_input_metadata"]["internal_all_args_leaves"] == 2
    assert metadata["lowered_input_metadata"]["internal_signature_equal"] is False
    assert "internal input signature" in metadata["signature_admission_error"]
    assert "mismatch_lower_seconds" in recorder.result["timing"]
    assert "mismatch_compile_seconds" not in recorder.result["timing"]


def test_actual_scalar_AOT_adapter_preserves_native_dtype_and_standard_rejection():
    from jax_orchidee.driver.global_batch import BlockArguments
    first=BlockArguments((np.ones((1,),dtype=np.float32),),np.array([2],dtype=np.float32),
        np.array([2],dtype=np.int32),(),(),(),(),(),())
    second=first._replace(forcing_days=np.array([3],dtype=np.float32))
    def scalar(values,forcing,*other): return tuple(v+forcing for v in values), {"forcing":forcing}
    recorder=SimpleNamespace(result={"timing":{}},phase=lambda name:None)
    executable=actual_base.lower_compile(scalar,first,'scalar',recorder)
    a,b=actual_base.run_two(executable,first,second)
    assert a[0][0].dtype==np.float32 and b[0][0].dtype==np.float32
    np.testing.assert_array_equal(b[0][0],np.array([6],dtype=np.float32))
    with pytest.raises(TypeError):
        executable(*first._replace(forcing_days=np.ones((2,),dtype=np.float32)))


def test_aot_signature_abstractifies_python_numpy_jax_and_aval_leaves():
    values = (1, np.float32(2), jax.numpy.asarray(3, dtype=np.int32),
              np.ones((2,), dtype=np.float32), jax.ShapeDtypeStruct((3,), np.float64))
    _, signature = actual_base._argument_signature(values)
    python_int_dtype = "int64" if jax.config.x64_enabled else "int32"
    assert signature == (
        ((), python_int_dtype, True), ((), "float32", False), ((), "int32", False),
        ((2,), "float32", False), ((3,), "float64", False),
    )
    aval = jax.core.ShapedArray((4,), np.dtype("float32"))
    assert actual_base._aval_signature((aval,)) == (((4,), "float32", False),)


def test_raw_python_scalar_lower_compile_and_bound_call_interfaces():
    recorder = SimpleNamespace(result={"timing": {}}, phase=lambda name: None)
    executable = actual_base.lower_compile(lambda value: value + 1.25, (2.5,), "python_float", recorder)
    assert recorder.result["compilations"]["python_float"]["lowered_input_signature"]["caller_leaves"] == 1
    assert float(executable(2.5)) == 3.75
    assert float(executable(jax.device_put(4.5))) == 5.75


def test_aot_audit_keeps_weak_type_strict():
    lowered = jax.jit(lambda value: value).lower(1)
    assert actual_base.audit_aot_signature(lowered, (1,), lowered=True)["public_tree_shape_dtype_weak_type_equal"]
    strong_same_dtype = np.asarray(1, dtype=np.dtype(actual_base._shaped_abstractify(1).dtype))
    assert actual_base._signature_leaf(strong_same_dtype)[:2] == actual_base._signature_leaf(1)[:2]
    assert actual_base._signature_leaf(strong_same_dtype)[2] is False
    with pytest.raises(TypeError, match="tree/shape/dtype/weak-type"):
        actual_base.audit_aot_signature(lowered, (strong_same_dtype,), lowered=True)


def test_scale_compile_batch_uses_shared_raw_leaf_safe_aot_path(monkeypatch):
    from jax_orchidee.driver.global_batch import BlockArguments, PreparedLane
    from scripts.dev import benchmark_global_scale as scale

    lanes, second = [], []
    for index in range(2):
        groups = ({"x": np.asarray([index], dtype=np.float64), "optional": None},)
        arguments = BlockArguments(groups, np.full((7, 48, 1), index, dtype=np.float64),
            np.arange(2, 9, dtype=np.int32), *({"x": np.asarray(index, dtype=np.float64)} for _ in range(6)))
        lanes.append(PreparedLane(str(index), None, None, None, None, arguments, (("science", 1),)))
        second.append(arguments._replace(forcing_days=arguments.forcing_days + 1))

    monkeypatch.setattr(scale.base.batch, "make_batch_transition", lambda selected: lambda *args: args[0])
    recorder = SimpleNamespace(result={"timing": {}}, phase=lambda name: None)
    executable, inputs = scale.compile_batch(lanes, second, "stubbed_physics", recorder)
    assert inputs[0].forcing_days.shape == (2, 7, 48, 1)
    assert executable(*inputs[0])[0]["x"].shape == (2, 1)
    assert recorder.result["compilations"]["stubbed_physics"]["lowered_input_signature"]["caller_leaves"] > 0


def sealed_capacity_fixture(tmp_path):
    """Tiny full-schema envelope exercises admission without a JIT or GPU."""
    source = tmp_path / "sealed"
    source.mkdir()
    identity = dict.fromkeys(("contract", "baseline_ref", "science_sha256", "lanes", "runtime",
        "compiler_environment", "warm_repeats", "gate", "precision_policy", "validation_mode",
        "workload_kind", "total_lanes", "lane_origins", "unique_real_points", "candidate_shapes",
        "conditional_512", "optional_B64_baseline", "conditional_policy", "numerical_reference_sha256",
        "continuation_B128_sha256"), "declared")
    measurements = {key: capacity.lane_metrics([1, 2, 3, 4, 5]) for key in ("core", "wrapper")}
    provenance = {"workload_kind": "capacity_screen_repeated_FULL64", "total_lanes": 512,
        "unique_real_points": 64, "chunk_size": 256, "chunks_per_collection": 2,
        "lane_days": 7168, "unique_point_days": 896, "blocks": [[2, 8], [9, 15]],
        "half_hours_per_day": 48, "full_outputs": True, "full_two_block_workload": True,
        "executable_memory_analysis": {"available": True, "temp_size_in_bytes": 123}}
    record = {"measurement_status": "complete", "strict_numerical_passed": True,
        "gates": [{"passed": True}] * 515, "timing": measurements, "provenance": provenance,
        "execution_origin": {"environment": {"git_head": "sealed-code", "loadavg": (1., 2., 3.)}}}
    outputs = tuple(np.zeros((512, 1), dtype=np.float32 if i % 2 else np.int32) for i in range(760))
    envelope = {"name": "capacity_B256_fixed512_v1", "status": "complete", "identity": identity,
        "value": {"outputs": outputs, "measurements": measurements}, "record": record}
    reference = tuple(value[:64] for value in outputs)
    return source, identity.copy(), envelope, reference


def import_fixture(source, identity, envelope, reference, *, manifest_change=None, envelope_change=None,
                   digest_change=False, byte_change=False):
    if envelope_change:
        envelope_change(envelope)
    data = pickle.dumps(envelope)
    digest = hashlib.sha256(data).hexdigest()
    record = {"status": "complete", "sha256": digest, **envelope["record"]}
    manifest = {"identity": envelope["identity"], "stages": {envelope["name"]: record}}
    if manifest_change:
        manifest_change(manifest)
    (source / "stage_capacity_B256_fixed512_v1.pkl").write_bytes(data)
    (source / "stage_manifest.json").write_text(json.dumps(manifest))
    return capacity._import_capacity_baseline(source, identity, source.parent / "new", jax,
        size=256, sha256="bad" if digest_change else digest,
        byte_count=len(data) + int(byte_change), job_id="test-job", code_sha="sealed-code", reference=reference)


def test_sealed_B256_import_keeps_native_schema_original_origin_and_actual_memory(tmp_path):
    source, identity, envelope, reference = sealed_capacity_fixture(tmp_path)
    timing, origin = import_fixture(source, identity, envelope, reference)
    assert timing == envelope["value"]["measurements"]
    assert origin["source_identity"] == identity
    assert origin["execution_origin"]["environment"]["loadavg"] == (1., 2., 3.)
    assert origin["provenance"]["executable_memory_analysis"]["temp_size_in_bytes"] == 123
    assert origin["role"] == "imported historical baseline; no new B256 measurement, executable, or destination stage"
    assert not (source.parent / "new").exists()


@pytest.mark.parametrize("key", ["science_sha256", "lanes", "runtime", "compiler_environment",
    "precision_policy", "numerical_reference_sha256", "continuation_B128_sha256", "candidate_shapes"])
def test_sealed_B256_import_rejects_identity_drift(tmp_path, key):
    source, identity, envelope, reference = sealed_capacity_fixture(tmp_path)
    identity[key] = "changed"
    with pytest.raises(ValueError, match=f"identity mismatch: {key}"):
        import_fixture(source, identity, envelope, reference)


@pytest.mark.parametrize("failure", ["checksum", "bytes", "failed-status", "origin", "gates",
    "native-dtype", "full-schema", "missing-wrapper", "memory", "timing", "loadavg-drift"])
def test_sealed_B256_import_rejects_changed_or_incomplete_evidence(tmp_path, failure):
    source, identity, envelope, reference = sealed_capacity_fixture(tmp_path)
    manifest_change = envelope_change = None
    if failure == "failed-status":
        def manifest_change(m):
            return m["stages"][envelope["name"]].update(status="failed")
    elif failure == "origin":
        def envelope_change(e):
            return e["record"]["execution_origin"]["environment"].update(git_head="new-code")
    elif failure == "gates":
        def envelope_change(e):
            return e["record"].update(gates=[{"passed": True}] * 514)
    elif failure == "native-dtype":
        def envelope_change(e):
            return e["value"].update(outputs=tuple(value.astype(np.float64) for value in e["value"]["outputs"]))
    elif failure == "full-schema":
        def envelope_change(e):
            return e["value"].update(outputs=list(e["value"]["outputs"]))
    elif failure == "missing-wrapper":
        def envelope_change(e):
            return e["value"]["measurements"].pop("wrapper")
    elif failure == "memory":
        def envelope_change(e):
            return e["record"]["provenance"]["executable_memory_analysis"].update(available=False)
    elif failure == "timing":
        def envelope_change(e):
            return e["value"]["measurements"]["core"].update(median_seconds=99)
    elif failure == "loadavg-drift":
        def manifest_change(m):
            return m["stages"][envelope["name"]]["execution_origin"]["environment"].update(loadavg=[1, 2, 4])
    with pytest.raises(ValueError):
        import_fixture(source, identity, envelope, reference, manifest_change=manifest_change,
            envelope_change=envelope_change, digest_change=failure == "checksum", byte_change=failure == "bytes")


@pytest.mark.parametrize("admitted", [True, False])
def test_B512_only_imports_origins_reuses_B256_memory_and_never_measures_baselines(monkeypatch, tmp_path, admitted):
    lanes, _, reference, _, recorder, scale, events = toy_case(monkeypatch)
    lanes = lanes[:64]
    second = [lane.arguments + 1 for lane in lanes]
    args = SimpleNamespace(output_dir=tmp_path / "new", full_reference_dir=tmp_path / "reference",
        memory_baseline_dir=tmp_path / "memory", prepared_input=tmp_path / "inputs/prepared.pkl",
        b128_baseline_dir=tmp_path / "old128", b256_baseline_dir=tmp_path / "old256",
        continue_512=True, continue_256_512=False, conditional_512=False, optional_b64_baseline=False,
        budget_seconds=2700, baseline_ref="source")
    old_memory = {"available": True, "temp_size_in_bytes": 10**10, "old_benchmark_peak_RSS_bytes": 100}
    b256_memory = {"available": True, "temp_size_in_bytes": 123}
    monkeypatch.setattr(capacity, "admit_numerical_sources", lambda *a: ({}, reference, {"sha256": "sealed"}, {"sha256": "old"}, old_memory))
    metrics = {key: capacity.lane_metrics([1]*5) for key in ("core", "wrapper")}
    monkeypatch.setattr(capacity, "import_b128_baseline", lambda *a, **kw: (metrics, {"role": "old128"}))
    def import256(directory, expected_identity, destination, jax, **kwargs):
        assert expected_identity["contract"] == capacity.CONTINUATION_CONTRACT
        assert expected_identity["continuation_B128_sha256"] == capacity.B128_SHA256
        return metrics, {"source_stage": "capacity_B256_fixed512_v1", "provenance": {"executable_memory_analysis": b256_memory}}
    monkeypatch.setattr(capacity, "import_b256_baseline", import256)
    scale.PRECISION_POLICY = "native"
    scale.screen_inputs = lambda *a: {}
    scale.base.legacy_factory = lambda *a: events.append(("legacy_factory", None))
    scale.base.roughheight_sentinel_contract = lambda *a: {}
    scale.jax.clear_caches = lambda: None
    recorder.phase = lambda *a, **kw: None
    recorder.flush = lambda: None
    if not admitted:
        scale.visible_gpu_memory = lambda: {"free_bytes": 1}
    calls = []
    def measure(size, args, recorder, *other):
        assert size == 512
        memories = other[-1]
        assert memories[256] == b256_memory
        assert capacity.memory_estimate(512, 100, 200, memories) == 200 + 600 + 246
        calls.append(size)
        return {"measurements": metrics}, {"measurement_status": "complete"}
    monkeypatch.setattr(capacity, "measure_shape", measure)
    if admitted:
        capacity.benchmark(args, recorder, lanes, second, scale)
        assert calls == [512] and recorder.result["completed_shapes"] == [512]
        assert recorder.result["comparison_shapes_including_imported_baseline"] == [128, 256, 512]
    else:
        with pytest.raises(MemoryError, match="B512-only resource admission"):
            capacity.benchmark(args, recorder, lanes, second, scale)
        assert calls == [] and recorder.result["completed_shapes"] == []
    manifest = json.loads((args.output_dir / "stage_manifest.json").read_text())
    assert manifest["identity"]["contract"] == capacity.B512_CONTRACT
    assert manifest["identity"]["budget_seconds"] == 2700
    assert "actual_science_source_sha256" in manifest["identity"]
    assert "execution_source_compatibility" in manifest["identity"]
    assert recorder.result["execution_source_identity"]["actual_science_source_sha256"] == manifest["identity"]["actual_science_source_sha256"]
    assert set(manifest["stages"]) == ({"capacity_B512_fixed512_v1"} if admitted else set())
    assert not any(event[0] == "compile" for event in events)


@pytest.mark.parametrize("copied_to_slurm_spool", [False, True])
def test_B512_launcher_actual_Bash_and_Python_dryrun_from_declared_checkout(tmp_path, copied_to_slurm_spool):
    root = Path(capacity.__file__).resolve().parents[2]
    bash = shutil.which("bash")
    resolved = subprocess.run([bash, "-c", 'cd "$1" && pwd', "path", str(tmp_path).replace("\\", "/")],
        text=True, capture_output=True, check=True).stdout.strip()
    declared = tmp_path / "runtime/worktrees/declared-code/scripts/hpc"
    declared.mkdir(parents=True)
    for name in ("slurm_global_gpu_capacity.sh", "slurm_global_gpu_capacity_512.sh"):
        source = (root / "scripts/hpc" / name).read_text().replace(
            "PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX", f"PROJECT={shlex.quote(resolved)}")
        (declared / name).write_text(source)
    launcher = declared / "slurm_global_gpu_capacity_512.sh"
    if copied_to_slurm_spool:
        spool = tmp_path / "slurm-spool/job-future"
        spool.mkdir(parents=True)
        launcher = spool / "slurm_script"
        shutil.copyfile(declared / "slurm_global_gpu_capacity_512.sh", launcher)
    env = os.environ.copy()
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES", "CONTINUE_512",
                "CONTINUE_256_512", "OPTIONAL_B64_BASELINE"):
        env.pop(key, None)
    env["CODE_DIR"] = resolved + "/runtime/worktrees/declared-code"
    result = subprocess.run([bash, str(launcher).replace("\\", "/"), "--dry-run"],
        env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "cpus=4 gpus=1 wall=00:45:00" in result.stdout
    tokens = shlex.split(next(line for line in result.stdout.splitlines() if line.startswith("command=")).removeprefix("command="))
    entry = next(i for i, token in enumerate(tokens) if token.endswith("global_batch_gpu_entry.py"))
    assert tokens[entry] == env["CODE_DIR"] + "/scripts/hpc/global_batch_gpu_entry.py"
    dry = subprocess.run([sys.executable, str(Path(capacity.__file__).resolve()), *tokens[entry + 1:], "--dry-run"],
        cwd=tmp_path, text=True, capture_output=True)
    assert dry.returncode == 0, dry.stderr
    report = json.loads(dry.stdout)
    assert report["candidates"] == [512] and report["budget_seconds"] == 2700
    assert report["contract"] == capacity.B512_CONTRACT
    assert report["JAX_imported"] is False and report["model_execution"] is False
    assert report["B512_requires_speedup"] is False and report["optional_B64"] is False
    assert "actual_science_source_sha256" in report["execution_source_identity"]
    assert not (tmp_path / "runtime/outputs").exists()


@pytest.mark.parametrize("extra,match", [
    (["--continue-512", "--continue-256-512"], "not allowed"),
    (["--continue-512", "--b128-baseline-dir", "old128", "--budget-seconds", "2700"], "sealed B256"),
    (["--b256-baseline-dir", "old256"], "requires B512-only"),
    (["--continue-512", "--b128-baseline-dir", "old128", "--b256-baseline-dir", "old256"], "2700s"),
    (["--continue-512", "--b128-baseline-dir", "old128", "--b256-baseline-dir", "old256",
      "--budget-seconds", "2700", "--optional-b64-baseline"], "forbids optional B64"),
])
def test_B512_rejects_mixed_or_incomplete_protocol_before_JAX(extra, match, tmp_path):
    command = [sys.executable, str(Path(capacity.__file__).resolve()), "--prepared-input", "inputs.pkl",
        "--full-reference-dir", "reference", "--memory-baseline-dir", "memory", "--output-dir", "new",
        "--dry-run", *extra]
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode != 0 and match in result.stderr


@pytest.mark.parametrize("change", ["helper-body", "mask-formula", "outside-helper", "altered-EOL"])
def test_source_certificate_rejects_every_unapproved_diffuco_byte_change(change):
    from scripts.dev import global_source_compatibility as certificate

    data = (Path(capacity.ROOT) / certificate.DIFFUCO_PATH).read_bytes()
    baseline, repaired = certificate.audit_diffuco_bytes(data)
    assert repaired is True
    assert certificate.audit_diffuco_bytes(baseline) == (baseline, False)
    if change == "helper-body":
        altered = data.replace(b"mask.setflags(write=False)", b"mask.setflags(write=True)")
    elif change == "mask-formula":
        altered = data.replace(b"mask = np.arange(int(nvm)) == int(pft_index)",
                               b"mask = np.arange(int(nvm)) <= int(pft_index)")
    elif change == "altered-EOL":
        altered = data.replace(b"\n", b"\r\n")
    else:
        altered = data + b"\n# unapproved module byte\n"
    assert altered != data
    with pytest.raises(ValueError, match="unapproved DIFFUCO"):
        certificate.audit_diffuco_bytes(altered)


@pytest.mark.parametrize("change", ["wrong-aggregate-formula", "altered-EOL"])
def test_source_certificate_rejects_unapproved_change_to_another_source_file(monkeypatch, tmp_path, change):
    from scripts.dev import global_source_compatibility as certificate

    approved = (Path(capacity.ROOT) / certificate.DIFFUCO_PATH).read_bytes()
    baseline, _ = certificate.audit_diffuco_bytes(approved)
    diffuco = tmp_path / certificate.DIFFUCO_PATH
    diffuco.parent.mkdir(parents=True)
    diffuco.write_bytes(approved)
    other_path = "jax_orchidee/scientific_other.py"
    other = tmp_path / other_path
    other.write_bytes(b"exact scientific bytes\n")
    paths = [certificate.DIFFUCO_PATH, other_path]
    digest = hashlib.sha256()
    for path, data in ((certificate.DIFFUCO_PATH, baseline), (other_path, other.read_bytes())):
        digest.update(path.encode() + b"\0" + data + b"\0")
    monkeypatch.setattr(certificate, "BASELINE_SHA256", digest.hexdigest())
    monkeypatch.setattr(certificate.subprocess, "check_output", lambda *args, **kwargs: "\n".join(paths))
    identity = certificate.source_identity(tmp_path)
    assert identity["science_sha256"] == digest.hexdigest()
    assert identity["actual_science_source_sha256"] != digest.hexdigest()
    other.write_bytes(b"changed scientific formula\n" if change == "wrong-aggregate-formula"
                     else b"exact scientific bytes\r\n")
    with pytest.raises(ValueError, match="outside certified"):
        certificate.source_identity(tmp_path)
