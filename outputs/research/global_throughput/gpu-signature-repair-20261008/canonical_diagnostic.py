import os, sys, json, hashlib, pickle, gc, argparse, subprocess
from pathlib import Path
os.environ.update(JAX_ENABLE_X64='true', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
parser=argparse.ArgumentParser()
parser.add_argument('--code-dir',type=Path,help='Import repaired helpers from this clean checkout instead of copied diagnostic definitions.')
options=parser.parse_args()
code_root=options.code_dir or Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v1')
sys.path.insert(0,str(code_root))
import jax, numpy as np
from types import SimpleNamespace
from scripts.dev.global_server_reference import _evidence_mismatch
jax.config.update('jax_enable_x64',True)
jax.config.update('jax_enable_compilation_cache',False)
def emit(**v): print(json.dumps(v,default=str),flush=True)
def _argument_signature(arguments):
    leaves, tree = jax.tree_util.tree_flatten((tuple(arguments), {}))
    return tree, tuple((tuple(value.shape), str(value.dtype), bool(getattr(value, "weak_type", False))) for value in leaves)

def _aval_signature(avals):
    return tuple((tuple(value.shape), str(value.dtype), bool(getattr(value, "weak_type", False))) for value in avals)

def audit_aot_signature(stage, arguments, *, lowered=False):
    """Fail before compile when public/internal AOT argument metadata diverges.

    The JAX0.4.38 private metadata is inspected diagnostically only; it is never
    rewritten. Public tree and dtype/shape admission remain mandatory.
    """
    tree, expected = _argument_signature(arguments)
    if stage.in_tree != tree or _aval_signature(jax.tree_util.tree_leaves(stage.in_avals)) != expected:
        raise TypeError("AOT public input tree/shape/dtype/weak-type signature differs from construction")
    mesh = stage._lowering if lowered else stage._executable
    info = mesh.compile_args.get("all_args_info") if lowered else getattr(mesh, "_all_args_info", None)
    kept = mesh.compile_args.get("kept_var_idx") if lowered else getattr(mesh, "_kept_var_idx", None)
    internal = None if info is None else _aval_signature(info.in_avals)
    if internal is not None and internal != expected:
        raise TypeError(f"AOT internal input signature differs: expected{len(expected)}, internal{len(internal)}")
    return {"caller_leaves": len(expected), "public_input_leaves": len(expected),
            "internal_all_args_leaves": None if internal is None else len(internal),
            "kept_input_leaves": None if kept is None else len(kept),
            "public_tree_shape_dtype_weak_type_equal": True,
            "internal_signature_equal": None if internal is None else True}

class BoundAOTExecutable:
    """Bind Python AOT dispatch to this stage's own immutable call parameters.

    JAX0.4.38 Compiled.__call__ lazily builds a C++ pjit callable. Calling the
    same stage's Compiled.call directly retains the standard tree, aval,
    sharding/layout and tracer checks without relying on that lazy callable.
    Physics, pruning, buffers and outputs are unchanged. This extra host
    dispatch overhead is included in all measured collection timings.
    """
    def __init__(self, compiled):
        self.compiled = compiled
        self.params = compiled._params

    def __call__(self, *arguments):
        from jax._src.stages import Compiled

        outputs, _, _ = Compiled.call(self.params, *arguments)
        return outputs

    def __getattr__(self, name):
        return getattr(self.compiled, name)

def lane_metrics(samples):
    if len(samples) != 5 or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("five positive fully synchronized samples required")
    median = statistics.median(samples)
    return {"samples_seconds": samples, "median_seconds": median, "range_seconds": [min(samples), max(samples)],
        "lane_days": TOTAL_LANES * 14, "lane_days_per_second": TOTAL_LANES * 14 / median,
        "unique_real_points": UNIQUE_POINTS, "unique_point_days": UNIQUE_POINTS * 14,
        "unique_point_days_is_throughput_denominator": False}

def import_b128_baseline(directory, expected_identity, destination, jax):
    """Validate sealed historical evidence without modifying or relabeling its stage."""
    from scripts.dev.global_batch_stage_store import StageStore

    StageStore.validate_source_destination(directory, destination)
    directory = Path(directory)
    manifest_path = directory / "stage_manifest.json"
    manifest_data = manifest_path.read_bytes()
    manifest = json.loads(manifest_data)
    source_identity = manifest.get("identity", {})
    admission_keys = ("contract", "baseline_ref", "science_sha256", "lanes", "runtime", "compiler_environment",
        "warm_repeats", "gate", "precision_policy", "validation_mode", "workload_kind", "total_lanes",
        "lane_origins", "unique_real_points", "candidate_shapes", "conditional_512", "optional_B64_baseline",
        "conditional_policy", "numerical_reference_sha256")
    for key in admission_keys:
        if key not in source_identity or source_identity[key] != expected_identity.get(key):
            raise ValueError(f"B128 baseline identity mismatch: {key}")
    name = "capacity_B128_fixed512_v1"
    record = manifest["stages"].get(name, {})
    if record.get("status") != "complete" or record.get("sha256") != B128_SHA256:
        raise ValueError("B128 baseline must be the sealed job15600405 stage")
    path = directory / ("stage_" + name + ".pkl")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != B128_SHA256 or len(data) != 220429366:
        raise ValueError("B128 baseline full checksum/bytes changed")
    envelope = pickle.loads(data)
    if (envelope.get("identity") != source_identity or envelope.get("name") != name
            or envelope.get("status") != "complete"):
        raise ValueError("B128 baseline envelope identity/status mismatch")
    proof, value = envelope["record"], envelope["value"]
    if (proof.get("measurement_status") != "complete" or proof.get("strict_numerical_passed") is not True
            or len(proof.get("gates", [])) != 515 or any(g.get("passed") is not True for g in proof["gates"])
            or proof.get("timing") != value.get("measurements") or proof.get("timing") != record.get("timing")
            or proof.get("provenance") != record.get("provenance")
            or _evidence_mismatch(proof.get("execution_origin"), record.get("execution_origin"),
                                  "record.execution_origin") is not None):
        raise ValueError("B128 baseline complete strict measurement/provenance mismatch")
    provenance = proof["provenance"]
    if any(provenance.get(key) != expected for key, expected in {
            "total_lanes": 512, "unique_real_points": 64, "chunk_size": 128,
            "blocks": [[2, 8], [9, 15]], "half_hours_per_day": 48,
            "full_outputs": True, "full_two_block_workload": True}.items()):
        raise ValueError("B128 baseline workload provenance mismatch")
    leaves = jax.tree_util.tree_leaves(value["outputs"])
    if len(leaves) != 760 or any(leaf.shape[0] != TOTAL_LANES for leaf in leaves):
        raise ValueError("B128 baseline full output schema mismatch")
    for metric in value["measurements"].values():
        if metric != {**metric, **lane_metrics(metric["samples_seconds"])}:
            raise ValueError("B128 baseline synchronized timing summary mismatch")
    origin = {"path": str(path), "sha256": B128_SHA256, "bytes": len(data),
        "manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
        "source_stage": name, "source_identity": source_identity, "execution_origin": proof["execution_origin"],
        "timing": value["measurements"], "provenance": provenance,
        "role": "imported historical baseline; no new B128 measurement, executable, or destination stage"}
    return value["measurements"], origin

import statistics, math
actual_base=None
if options.code_dir is not None:
    from scripts.dev import benchmark_global_batch as actual_base
    from scripts.dev import benchmark_global_gpu_capacity as actual_capacity
    audit_aot_signature=actual_base.audit_aot_signature
    BoundAOTExecutable=actual_base.BoundAOTExecutable
    import_b128_baseline=actual_capacity.import_b128_baseline
    lane_metrics=actual_capacity.lane_metrics
    emit(phase='actual_module_identity',git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code_root,text=True).strip(),
         helpers=[{'path':str(Path(m.__file__).resolve()),'sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()} for m in (actual_base,actual_capacity)])
B128_SHA256='e91db1391ccb5f525f9ec29f1cba24235c4ce9cd4153b9297fe2ed014d0c9cb5'
TOTAL_LANES=512
UNIQUE_POINTS=64
origin=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v1-load512/capacity')
identity=json.loads((origin/'stage_manifest.json').read_text())['identity']
timing, proof=import_b128_baseline(origin,identity,Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/gpu-signature-repair-20261008'),jax)
emit(phase='sealed_baseline_validated',sha256=proof['sha256'],bytes=proof['bytes'],core=timing['core']['median_seconds'],wrapper=timing['wrapper']['median_seconds'],execution_origin_preserved=True)
for key in ('science_sha256','lanes'):
    try: import_b128_baseline(origin,{**identity,key:'changed'},Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/gpu-signature-repair-20261008'),jax)
    except ValueError as exc: emit(phase='baseline_changed_identity_rejected',key=key,message=str(exc))
    else: raise AssertionError(key)
counts=[350,14,1,4,10,20,60,38,39]
# 536 leaves in9 groups,417 active; nested673-leaf signature adds137 unused.
for size in (128,256,512):
    groups=tuple(tuple(np.full((size,1),i+j*.01,dtype=np.float64) for j in range(c)) for i,c in enumerate(counts))
    def transition(*groups):
        flat=jax.tree_util.tree_leaves(groups)
        return tuple(flat[:417])
    nested_args=groups+(tuple(np.zeros((size,1)) for _ in range(137)),)
    def nested(*groups): return transition(*groups[:9])
    nested_compiled=jax.jit(nested).lower(*nested_args).compile()
    lowered=jax.jit(transition).lower(*groups)
    before=audit_aot_signature(lowered,groups,lowered=True)
    if actual_base is None:
        compiled=lowered.compile()
        bound=BoundAOTExecutable(compiled)
    else:
        recorder=SimpleNamespace(result={'timing':{}},phase=lambda name:None)
        bound=actual_base.lower_compile(transition,groups,'tiny_B'+str(size),recorder)
        compiled=bound.compiled
    after=audit_aot_signature(compiled,groups)
    normal=compiled(*groups)
    bound_result=bound(*groups)
    for a,b in zip(normal,bound_result,strict=True): np.testing.assert_array_equal(a,b)
    # Fault injection isolates lazycall misbinding and verifies ownparameter binding.
    compiled._call=nested_compiled._executable.create_cpp_call(compiled._no_kwargs,compiled.in_tree,compiled.out_tree)
    try: compiled(*groups)
    except TypeError as exc:
        assert 'compiledfor673inputsbutcalledwith536' in str(exc).replace(' ','')
        error=str(exc)
    else: raise AssertionError('misbound lazy callable unexpectedly succeeded')
    corrected=bound(*groups)
    for a,b in zip(normal,corrected,strict=True): np.testing.assert_array_equal(a,b)
    emit(phase='sequential_AOT',shape=size,lowered=before,compiled=after,active417=True,misbinding_fault_error=error,bound_call_exact=True,natural_failure_reproduced=False)
    del nested_compiled,lowered,compiled,bound,normal,bound_result,corrected
    jax.clear_caches(); gc.collect()
emit(phase='complete',jax=jax.__version__,host=os.uname().nodename,scope='tiny CPU surrogate+sealed baseline read; no real model compile/execution')

