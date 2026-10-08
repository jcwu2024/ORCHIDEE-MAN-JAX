"""Capacity protocol tests with tiny NumPy outputs; no physics or JIT compile."""
from __future__ import annotations

import json
import os
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
    recorder = SimpleNamespace(result={"gates": [], "timing": {}, "server_resources": {}, "environment": {}}, sentinel_paths={})
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
    monkeypatch.setattr(capacity,'import_b128_baseline',lambda *a: (baseline,{"role":"historical"}))
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
    for got,want in zip(a[0],expected_a,strict=True): np.testing.assert_array_equal(got,want)
    for got,want in zip(b[0],expected_b,strict=True): np.testing.assert_array_equal(got,want)
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
