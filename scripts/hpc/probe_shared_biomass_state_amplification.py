"""CPU scalar forward capture of actual source-kernel biomass ratio boundaries.

This probe reconstructs a forward from declared source overlays and optionally
evaluates the public physical-tail loss AD with captured neural outputs fixed.
An explicit hash-checked single-function overlay can compare source formulas
in memory. All arrays are consolidated in one small NPZ; the immutable checkout
is preserved. The scope excludes historical CUDA replay and neural state AD.
"""
from __future__ import annotations

import time
STARTED = time.monotonic()
import argparse
import ast
from contextlib import contextmanager, ExitStack
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading

import numpy as np

SCHEMA = "shared_biomass_actual_boundary_capture_v1"
KERNELS = ("npp_closed_update", "npp_leaf_age_sla_age_update", "turnover_leaf_age_fall", "_stable_ratio_for_ad")
_PREPARED_CACHE = {}
_NEURAL_EXECUTABLE_CACHE = {}


def adjacent(name):
    root = os.environ.get("ORCHIDEE_REPO_ROOT", str(Path(__file__).resolve().parents[2]))
    if root not in sys.path: sys.path.insert(0, root)
    canonical = "scripts.hpc."+name
    path = Path(__file__).with_name(name+".py")
    module = sys.modules.get(canonical)
    if module is not None and Path(module.__file__).resolve() == path.resolve(): return module
    spec = importlib.util.spec_from_file_location(canonical, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[canonical] = module
    spec.loader.exec_module(module)
    return module


def read_pickle(path, *, reuse=False):
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if reuse and key in _PREPARED_CACHE: return _PREPARED_CACHE[key]
    with path.open("rb") as handle: result = pickle.load(handle)
    if reuse: _PREPARED_CACHE[key] = result
    return result


def array_summary(value):
    a = np.asarray(value)
    finite = np.isfinite(a)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "coordinates": int(a.size),
        "finite": bool(np.all(finite)), "zero": int(np.count_nonzero(a == 0)),
        "minimum": float(np.min(a[finite])) if np.any(finite) else None,
        "maximum": float(np.max(a[finite])) if np.any(finite) else None,
        "maximum_absolute": float(np.max(np.abs(a[finite]))) if np.any(finite) else None}


class BoundaryCollector:
    """Capture concrete actual arguments/results, never substitute state inputs."""
    def __init__(self, *, max_bytes=32*2**20, on_boundary=None):
        self.arrays, self.records, self.stack = {}, [], []
        self.max_bytes, self.total_bytes, self.on_boundary = max_bytes, 0, on_boundary
        self.lock = threading.RLock()

    def put(self, key, value):
        # Tracers mean the proposed eager capture did not observe execution.
        # Reject instead of materializing an invented/initial-state surrogate.
        if type(value).__module__.startswith("jax") and "Tracer" in type(value).__name__:
            raise RuntimeError("actual kernel boundary is a Tracer; eager capture unavailable")
        try: array = np.asarray(value)
        except Exception as error: raise RuntimeError("actual boundary cannot be materialized") from error
        if array.dtype.kind not in "biufc": raise TypeError("boundary must be numeric/bool")
        array = np.array(array, copy=True)
        with self.lock:
            if self.total_bytes+array.nbytes > self.max_bytes:
                raise MemoryError("bounded capture array budget exceeded")
            self.total_bytes += array.nbytes
            self.arrays[key] = array
        return {"array": key, **array_summary(array)}

    def begin(self, name, function, args, kwargs):
        bound = inspect.signature(function).bind(*args, **kwargs)
        bound.apply_defaults()
        with self.lock:
            index = len(self.records)
            record = {"index": index, "symbol": name, "parent": self.stack[-1] if self.stack else None,
                      "status": "entered", "inputs": {}, "outputs": {},
                      "entered_elapsed_seconds": time.monotonic()-STARTED}
            self.records.append(record)
            if hasattr(self, "active_day"): record["rollout_day"] = self.active_day
        for key, value in bound.arguments.items():
            record["inputs"][key] = ({"literal": None} if value is None
                else self.put(f"call{index}.input.{key}", value))
        self.stack.append(index)
        return record, bound.arguments

    def finish(self, record, result):
        import jax
        leaves, treedef = jax.tree_util.tree_flatten(result)
        record["output_treedef"] = str(treedef)
        for index, value in enumerate(leaves):
            record["outputs"][str(index)] = self.put(f"call{record['index']}.output.{index}", value)
        record["status"] = "returned"
        record["returned_elapsed_seconds"] = time.monotonic()-STARTED
        record["seconds_including_nested_capture"] = record["returned_elapsed_seconds"]-record["entered_elapsed_seconds"]

    def wrap(self, name, function):
        def observed(*args, **kwargs):
            record, arguments = self.begin(name, function, args, kwargs)
            try:
                result = function(*args, **kwargs)
                self.finish(record, result)
                if name == "_stable_ratio_for_ad":
                    denominator = np.asarray(arguments["denominator"])
                    numerator = np.asarray(arguments["numerator"])
                    cutoff = np.sqrt(np.finfo(denominator.dtype).tiny)
                    record["ratio_branch_diagnostics"] = {
                        "denominator_nonzero": self.put(f"call{record['index']}.denominator_nonzero", denominator != 0),
                        "tangent_representable": self.put(f"call{record['index']}.tangent_representable", np.abs(denominator) >= cutoff),
                        "denominator_positive": self.put(f"call{record['index']}.denominator_positive", denominator > 0),
                        "numerator_zero": self.put(f"call{record['index']}.numerator_zero", numerator == 0),
                        "source_tangent_cutoff": float(cutoff),
                        "note": "actual ratio call operands after caller masking; not initial biomass"}
                if name in ("jnp.minimum", "jnp.maximum"):
                    x, y = np.asarray(arguments["x"]), np.asarray(arguments["y"])
                    record["comparison_diagnostics"] = {
                        relation: self.put(f"call{record['index']}.{relation}", value)
                        for relation, value in (("tie", x == y), ("x_less_y", x < y), ("x_greater_y", x > y))}
                return result
            except Exception as error:
                record.update(status="error", error_type=type(error).__name__, error=str(error))
                raise
            finally:
                self.stack.pop()
                # Nested where/min/max remain in memory and are included in
                # the next ratio/kernel flush and watchdog flush. Rewriting
                # the full neural NPZ for every elementary op is unnecessary.
                if self.on_boundary and (not self.stack or name == "_stable_ratio_for_ad"):
                    self.on_boundary()
        return observed


def capture_tree(collector, prefix, tree):
    """Consolidate public diagnostic leaves; JSON contains only array metadata."""
    import jax
    leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    return {"treedef": str(treedef), "leaves": [
        {"path": jax.tree_util.keystr(path), **collector.put(f"{prefix}.{index}", value)}
        for index, (path, value) in enumerate(leaves)]}


def biomass_layout(contract, coordinate=2243):
    """Resolve the suspicious coordinate from the actual prepared contract."""
    matches = [leaf for leaf in contract.state_leaves if leaf.key.endswith(".biomass")]
    if len(matches) != 1: raise ValueError("one actual biomass state leaf required")
    leaf = matches[0]
    if leaf.start is None or leaf.stop is None or not leaf.start <= coordinate < leaf.stop:
        raise ValueError("suspicious coordinate is outside actual biomass binding")
    if int(np.prod(leaf.shape)) != leaf.stop-leaf.start: raise ValueError("biomass shape/span differs")
    local = coordinate-leaf.start
    return {"key": leaf.key, "start": leaf.start, "stop": leaf.stop,
        "shape": list(leaf.shape), "axis_names": list(leaf.axis_names),
        "selected_pft_indices": list(leaf.selected_pft_indices), "source_ref": leaf.source_ref,
        "coordinate": coordinate, "local_coordinate": local,
        "unravelled_coordinate": [int(value) for value in np.unravel_index(local, leaf.shape)]}


def timed_neural_component(function, name, execution, emit, *, capture=None):
    """Observe neural progress; optionally JIT only this unchanged forward."""
    import jax
    compiled = {}
    def call(*args, **kwargs):
        started = time.monotonic()
        emit(name+"_entered", execution=execution)
        if execution == "component_jit":
            # Schema/layout/config and frozen normalization are unchanged
            # closed metadata. Parameters/context/current inputs stay dynamic.
            source_path = Path(inspect.getsourcefile(function))
            source_stat = source_path.stat()
            key = (function.__module__, function.__name__, str(source_path), source_stat.st_mtime_ns,
                source_stat.st_size, tuple((key, id(value)) for key, value in sorted(kwargs.items())))
            if key in _NEURAL_EXECUTABLE_CACHE:
                compiled[key] = _NEURAL_EXECUTABLE_CACHE[key]
                emit(name+"_same_process_executable_reused")
            with jax.disable_jit(False):
                if key not in compiled:
                    began = time.monotonic()
                    lowered = jax.jit(lambda *dynamic: function(*dynamic, **kwargs)).lower(*args)
                    emit(name+"_lowered", seconds=time.monotonic()-began)
                    began = time.monotonic()
                    compiled[key] = lowered.compile()
                    _NEURAL_EXECUTABLE_CACHE[key] = compiled[key]
                    emit(name+"_compiled", seconds=time.monotonic()-began)
                result = compiled[key](*args)
        else:
            result = function(*args, **kwargs)
        jax.block_until_ready(result)
        emit(name+"_returned", seconds=time.monotonic()-started)
        if capture: capture(result)
        return result
    return call


def validate_neural_outputs(outputs, head):
    """Validate actual saved forward ABI before freezing any neural output."""
    raw, masks = outputs["actual_neural_raw_outputs"]
    expected = {field.name: tuple(field.shape) for field in head.learned_schema.fields}
    if set(raw) != set(expected) or set(masks) != set(expected):
        raise ValueError("captured neural output field registry differs")
    for name, shape in expected.items():
        for tree, dtype in ((raw, "float32"), (masks, "bool")):
            array = np.asarray(tree[name])
            if array.shape != shape or array.dtype != np.dtype(dtype) or not np.all(np.isfinite(array)):
                raise ValueError("captured neural field shape/dtype/finite differs: "+name)
    context = outputs["actual_neural_context"]
    width = head.shared_config.width
    experts = int(np.max(head.layout.field_expert_index))+1
    pfts = len(head.layout.active_pft_indices)
    source_count = np.shape(context.source_tokens)[0]
    axes = np.shape(context.expert_axis_tokens)[1]
    expected_context = {
        "source_tokens": ((source_count, width), "float32"), "source_mask": ((source_count,), "bool"),
        "source_family_index": ((source_count,), "int32"),
        "expert_axis_tokens": ((experts, axes, width), "float32"), "expert_axis_mask": ((experts, axes), "bool"),
        "pft_tokens": ((pfts, width), "float32"), "pft_mask": ((pfts,), "bool"),
        "process_tokens": ((experts, width), "float32"), "process_source_mask": ((experts, source_count), "bool")}
    if tuple(context._fields) != tuple(expected_context): raise ValueError("captured neural context field ABI differs")
    for name, (shape, dtype) in expected_context.items():
        array = np.asarray(getattr(context, name))
        if array.shape != shape or array.dtype != np.dtype(dtype) or not np.all(np.isfinite(array)):
            raise ValueError("captured neural context shape/dtype/finite differs: "+name)
    return {"fields": len(expected), "context_shape_dtype_finite": True, "raw_shape_dtype_finite": True}


@contextmanager
def install_turnover_overlay(module, path, expected_base_sha256):
    """Replace one verified source symbol in memory, never edit the checkout."""
    original = module.turnover_leaf_age_fall
    base_sha = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    if base_sha != expected_base_sha256:
        raise ValueError("immutable turnover baseline function source hash differs")
    source = Path(path).read_text(encoding="utf-8")
    parsed = ast.parse(source)
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.FunctionDef) or parsed.body[0].name != "turnover_leaf_age_fall":
        raise ValueError("diagnostic turnover overlay must contain exactly the one source function")
    namespace = dict(module.__dict__)
    exec(compile(source, str(path), "exec"), namespace)
    replacement = namespace["turnover_leaf_age_fall"]
    if inspect.signature(replacement) != inspect.signature(original):
        raise ValueError("diagnostic turnover overlay signature differs")
    replaced = []
    for loaded in tuple(sys.modules.values()):
        bindings = getattr(loaded, "__dict__", None)
        if bindings is None: continue
        for key, value in tuple(bindings.items()):
            if value is original:
                replaced.append((bindings, key, value))
                bindings[key] = replacement
    try:
        yield {"path": str(path), "sha256": hashlib.sha256(source.encode()).hexdigest(),
               "verified_base_function_sha256": base_sha, "checkout_edited": False}
    finally:
        for bindings, key, value in reversed(replaced): bindings[key] = value


@contextmanager
def observe_turnover_output_cotangent(module, callback):
    """Native physical VJP boundary observation; may reassociate FP AD work."""
    import jax
    original = module.turnover_leaf_age_fall
    def observed(*args, **kwargs):
        @jax.custom_vjp
        def operation(payload): return original(*payload[0], **payload[1])
        def forward(payload): return original(*payload[0], **payload[1]), payload
        def backward(payload, cotangent):
            leaves, structure = jax.tree_util.tree_flatten(cotangent)
            jax.debug.callback(lambda *values: callback(jax.tree_util.tree_unflatten(structure, values)),
                *leaves, ordered=True)
            _, pullback = jax.vjp(lambda data: original(*data[0], **data[1]), payload)
            return (pullback(cotangent)[0],)
        operation.defvjp(forward, backward)
        return operation((args, kwargs))
    replaced = []
    for loaded in tuple(sys.modules.values()):
        bindings = getattr(loaded, "__dict__", None)
        if bindings is None: continue
        for key, value in tuple(bindings.items()):
            if value is original:
                replaced.append((bindings, key, value))
                bindings[key] = observed
    try: yield
    finally:
        for bindings, key, value in reversed(replaced): bindings[key] = value


@contextmanager
def instrument_source_kernels(collector, module):
    """Patch actual aliases too; restore every binding after this CPU probe."""
    originals = {name: getattr(module, name) for name in KERNELS}
    wrappers = {id(function): collector.wrap(name, function) for name, function in originals.items()}
    replaced = []
    namespaces = [getattr(loaded, "__dict__", None) for loaded in tuple(sys.modules.values())]
    namespaces += [function.__globals__ for function in originals.values() if hasattr(function, "__globals__")]
    seen = set()
    for namespace in namespaces:
        if namespace is None: continue
        if id(namespace) in seen: continue
        seen.add(id(namespace))
        for key, value in tuple(namespace.items()):
            replacement = wrappers.get(id(value))
            if replacement is not None:
                replaced.append((namespace, key, value))
                namespace[key] = replacement
    # Observe actual predicates and min/max ties only while inside a selected
    # physical kernel. The neural network and loss retain their normal calls.
    import jax.numpy as jnp
    for name in ("where", "minimum", "maximum"):
        original = getattr(jnp, name)
        if name in ("minimum", "maximum"):
            def binary(function):
                def call(x, y): return function(x, y)
                return call
            target = binary(original)
        else: target = original
        observed = collector.wrap("jnp."+name, target)
        def selected(*args, _original=original, _observed=observed, **kwargs):
            return _observed(*args, **kwargs) if collector.stack else _original(*args, **kwargs)
        replaced.append((jnp.__dict__, name, original))
        setattr(jnp, name, selected)
    try: yield
    finally:
        for namespace, key, value in reversed(replaced): namespace[key] = value


def select_actual_sample(jax, bundle, artifact_id, row):
    ids = list(bundle["artifact_ids"])
    if ids.count(artifact_id) != 1 or not 0 <= row < 32:
        raise ValueError("one declared artifact and daily row0..31 required")
    payload = bundle["payloads"][ids.index(artifact_id)]
    daily = payload["daily"]
    sample = jax.tree_util.tree_map(lambda value: value[row], daily)
    if sample.inputs.continuous_state.shape != (3854,): raise ValueError("actual scalar state width differs")
    chunk_start = (row//4)*4
    digest = hashlib.sha256()
    for value in jax.tree_util.tree_leaves(daily):
        array = np.ascontiguousarray(value[chunk_start:chunk_start+4])
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return sample, {"artifact_id": artifact_id, "artifact_index": ids.index(artifact_id),
        "row": row, "micro4_row_start": chunk_start, "micro4_row_stop": chunk_start+4,
        "micro4_sha256": digest.hexdigest(), "source": "actual compact input bundle"}


def validate_snapshot(jax, saved, manifest, schedule):
    expected_manifest = {k: v for k, v in manifest.items() if k != "resume"}
    actual_manifest = {k: v for k, v in saved.get("manifest", {}).items() if k != "resume"}
    step = manifest["resume"]["completed_pairs"]
    if (saved.get("scope") != "original_full_state_schedule_diagnostic_not_formal_training"
            or saved.get("schedule") != schedule or actual_manifest != expected_manifest
            or saved.get("completed_diagnostic_updates") != step
            or saved.get("adoption") is not False or saved.get("scientific_acceptance") is not False):
        raise ValueError("capture requires exact admitted independent saved chain/provenance")
    parameters, optimizer = saved["parameters"], saved["optimizer"]
    leaves = jax.tree_util.tree_leaves(parameters)
    if len(leaves) != 300 or sum(np.size(x) for x in leaves) != 64884481:
        raise ValueError("capture original full parameter tree differs")
    for tree in (parameters, optimizer.first, optimizer.second):
        if jax.tree_util.tree_structure(tree) != jax.tree_util.tree_structure(parameters):
            raise ValueError("capture AdamW tree differs")
        for value, parameter in zip(jax.tree_util.tree_leaves(tree), leaves, strict=True):
            if np.shape(value) != np.shape(parameter) or np.asarray(value).dtype != np.dtype("float32") or not np.all(np.isfinite(value)):
                raise ValueError("capture parameter/AdamW shape/dtype/finite differs")
    if (np.shape(optimizer.step) != () or np.asarray(optimizer.step).dtype != np.dtype("int32")
            or int(np.asarray(optimizer.step)) != step): raise ValueError("capture public AdamW step differs")
    if any(np.any(np.asarray(x) < 0) for x in jax.tree_util.tree_leaves(optimizer.second)):
        raise ValueError("capture AdamW second moments invalid")
    return parameters, step


def persist_capture(output, report, collector):
    from scripts.hpc.training_artifact_io import _atomic_json
    npz = output.with_suffix(".npz")
    temporary = npz.with_name(f".{npz.name}.{os.getpid()}.tmp")
    with collector.lock:
        with temporary.open("wb") as handle: np.savez_compressed(handle, **collector.arrays)
        os.replace(temporary, npz)
        report.update(boundaries=collector.records, array_artifact=str(npz),
                      capture_array_bytes=collector.total_bytes, elapsed_seconds=time.monotonic()-STARTED)
        _atomic_json(output, report)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "prepared-context", "chain-snapshot", "field-adapter", "backbone-adapter", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--schedule", choices=("old", "new"), default="old")
    p.add_argument("--artifact-id", type=int, default=275)
    p.add_argument("--row", type=int, default=17)
    p.add_argument("--reference-state-diagnostics", type=Path)
    p.add_argument("--reference-step", type=int, default=2)
    p.add_argument("--wall-seconds", type=float, default=180)
    p.add_argument("--neural-execution", choices=("eager", "component_jit"), default="eager")
    p.add_argument("--neural-cache", type=Path, help="reuse this probe's exact already captured neural outputs only")
    p.add_argument("--physical-state-vjp", action="store_true",
        help="optional CPU scalar physical-tail state VJP with exact cached neural outputs held constant")
    p.add_argument("--physical-state-vjp-repeats", type=int, choices=(1, 2), default=1,
        help="same-process repeat with warm physical primitives; unchanged frozen neural cache")
    p.add_argument("--additional-capture", action="append", choices=("old:16", "old:17", "new:16", "new:17"), default=[],
        help="bounded same-process extra captures with own exact schedule snapshot and output/cache identity")
    p.add_argument("--turnover-overlay", type=Path)
    p.add_argument("--turnover-base-sha256", help="required exact immutable baseline function-source SHA for overlay")
    p.add_argument("--capture-turnover-cotangent", action="store_true",
        help="observe actual physical-loss turnover output cotangent through a disclosed custom VJP boundary")
    return p


def run(args):
    global STARTED
    STARTED = time.monotonic()
    if not 10 <= args.wall_seconds <= 600: raise ValueError("bounded CPU wall10..600s required")
    if args.physical_state_vjp and not args.neural_cache:
        raise ValueError("bounded physical-state VJP requires exact complete captured neural outputs")
    if bool(args.turnover_overlay) != bool(args.turnover_base_sha256):
        raise ValueError("turnover overlay requires its exact declared immutable baseline function hash")
    if args.capture_turnover_cotangent and not args.physical_state_vjp:
        raise ValueError("actual output cotangent capture requires physical-state VJP")
    if any(args.output.with_suffix(suffix).exists() for suffix in (".json", ".npz", ".neural.pkl")): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Backend selection precedes any prepared object unpickling/import.
    os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    import jax
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "cpu": raise ValueError("capture requires CPU only")
    source = adjacent("probe_shared_original_full_state_schedule")
    adjacent("shared_full_state_ad_execution")
    adjacent("training_artifact_io")
    owner = adjacent("probe_shared_expert_real_daily")
    manifest = source.read_manifest(args.manifest)
    if "resume" not in manifest or str(args.chain_snapshot) != manifest["resume"][args.schedule]:
        raise ValueError("capture requires the explicitly admitted resume chain path")
    profile, reference = source.read_reference(manifest)
    resume = source.validate_resume_metadata(manifest, reference)
    source.validate_original_assets(manifest, reference)
    root = Path(os.environ.get("ORCHIDEE_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != reference["source_git_head"]: raise ValueError("capture checkout differs from source provenance")
    report = {"schema_version": SCHEMA, "status": "running", "scope": "reconstructed_CPU_scalar_forward_actual_boundaries",
        "artifact_id": args.artifact_id, "row": args.row, "schedule": args.schedule,
        "source_git_head": head, "prepared_context": str(args.prepared_context), "chain_snapshot": str(args.chain_snapshot),
        "GPU_executed": False, "network_AD": False, "full_network_gradient_compilations": 0,
        "precision": "unchanged_FP32_neural_FP64_physical", "jax_disable_jit": True,
        "neural_execution": args.neural_execution, "stage_events": [],
        "scientific_acceptance": False, "adoption": False, "historical_binary_cause_established": False,
        "source_correspondence": "reconstructed declared overlays; source hashes checked below; no CUDA binary replay",
        "prior_strict_parity_passed": resume["prior_strict_parity_passed"]}
    collector = BoundaryCollector(on_boundary=lambda: persist_capture(args.output, report, collector))
    def timeout():
        report.update(status="incomplete_timeout")
        try: persist_capture(args.output, report, collector)
        finally: os._exit(124)
    timer = threading.Timer(max(.01, args.wall_seconds-(time.monotonic()-STARTED)), timeout)
    timer.daemon = True
    timer.start()
    source_patches = ExitStack()
    try:
        report["stage"] = "load_actual_inputs_and_chain"
        persist_capture(args.output, report, collector)
        context = read_pickle(args.prepared_context, reuse=True)
        expected = owner.context_identity(reference, Path(manifest["original_checkpoint"]), reference["method"]["training_plan"]["training_artifact_ids"][0])
        if (context.get("identity") != expected or Path(context["daily_executable"]).name != Path(profile["executables"]["daily"]["executable"]).name):
            raise ValueError("capture prepared context provenance differs")
        bundle = read_pickle(manifest["input_bundle"], reuse=True)
        workload = source.validate_bundle(jax, bundle, manifest, reference)
        sample, binding = select_actual_sample(jax, bundle, args.artifact_id, args.row)
        if binding["micro4_sha256"] != workload["daily_micro4_sha256"][binding["artifact_index"]][args.row//4]:
            raise ValueError("capture micro4 binding differs")
        saved = read_pickle(args.chain_snapshot)
        parameters, step = validate_snapshot(jax, saved, manifest, args.schedule)
        report.update(input_binding=binding, parameter_version=step)
        del saved, bundle
        collector.put("actual_scalar_initial_state", sample.inputs.continuous_state)
        report["biomass_layout"] = biomass_layout(context["head"].contract)
        report["actual_input_metadata"] = capture_tree(collector, "actual_input_metadata", {
            "pft": sample.inputs.pft, "year": sample.inputs.year, "day_index": sample.inputs.day_index,
            "discrete_state": sample.inputs.discrete_state})
        report["actual_process_targets"] = capture_tree(collector, "actual_process_targets", sample.process_targets)
        report["actual_expected_next_state"] = capture_tree(collector, "actual_expected_next_state", {
            "state": sample.expected_next_state, "defined": sample.expected_next_state_defined})
        admission = json.loads(Path(manifest["components"]["daily"]["report"]).read_text())
        report["overlay_source_hashes"] = {}
        for key, path in (("field_expert_adapter", args.field_adapter), ("backbone_adapter", args.backbone_adapter)):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            expected_hash = admission["source_artifacts"][key]["sha256"]
            report["overlay_source_hashes"][key] = {"path": str(path), "sha256": digest,
                "historic_sha256": expected_hash, "matches_historic": digest == expected_hash}
        adapter = owner.load_staged_adapter(args.field_adapter)
        assembler = owner.load_backbone_assembler(args.backbone_adapter)
        cache_identity = {key: report[key] for key in ("source_git_head", "prepared_context", "chain_snapshot",
            "schedule", "parameter_version", "input_binding", "overlay_source_hashes")}
        neural_cache = {"schema_version": "actual_CPU_forward_neural_cache_v1", "identity": cache_identity,
            "captured_neural_execution": args.neural_execution, "outputs": {}}
        if args.neural_cache:
            neural_cache = read_pickle(args.neural_cache)
            if (neural_cache.get("schema_version") != "actual_CPU_forward_neural_cache_v1"
                    or neural_cache.get("identity") != cache_identity
                    or set(neural_cache.get("outputs", {})) != {"actual_neural_context", "actual_neural_raw_outputs"}):
                raise ValueError("actual captured neural cache identity/completeness differs")
            report["neural_cache_reused"] = str(args.neural_cache)
            report["captured_neural_execution"] = neural_cache["captured_neural_execution"]
            report["neural_cache_ABI_validation"] = validate_neural_outputs(neural_cache["outputs"], context["head"])
        def emit(stage, **metadata):
            report["stage"] = stage
            report["stage_events"].append({"stage": stage, "elapsed_seconds": time.monotonic()-STARTED, **metadata})
            persist_capture(args.output, report, collector)
        def capture_neural(name, result):
            report[name] = capture_tree(collector, name, result)
            neural_cache["outputs"][name] = jax.device_get(result)
            cache_path = args.output.with_suffix(".neural.pkl")
            temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
            with temporary.open("wb") as handle: pickle.dump(neural_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, cache_path)
            report["actual_neural_cache_artifact"] = str(cache_path)
            emit(name+"_captured")
        def neural(function, stage, key):
            if args.neural_cache:
                def replay(*_args, **_kwargs):
                    emit(stage+"_actual_cache_replayed")
                    result = jax.device_put(neural_cache["outputs"][key])
                    capture_neural(key, result)
                    return result
                return replay
            return timed_neural_component(function, stage, args.neural_execution, emit,
                capture=lambda result: capture_neural(key, result))
        adapter = neural(adapter, "neural_decoder", "actual_neural_raw_outputs")
        assembler = neural(assembler, "neural_backbone", "actual_neural_context")
        from research.daily_coarse_graining.daily_operator import shared_process_head
        from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_joint_daily_objective
        from jax_orchidee.stomate import carbon_kernels
        if args.turnover_overlay:
            report["diagnostic_turnover_overlay"] = source_patches.enter_context(
                install_turnover_overlay(carbon_kernels, args.turnover_overlay, args.turnover_base_sha256))
        objective = make_shared_joint_daily_objective(context["runtime"], context["head"], context["template"],
            context["state_scales"], context["science_scales"], include_state_diagnostics=True)
        report["kernel_sources"] = {name: {"file": inspect.getsourcefile(getattr(carbon_kernels, name)),
            "line": inspect.getsourcelines(getattr(carbon_kernels, name))[1]} for name in KERNELS}
        report["stage"] = "scalar_forward_no_network_AD"
        persist_capture(args.output, report, collector)
        forward_started = time.monotonic()
        with jax.disable_jit(), owner.candidate_process_head_patches(shared_process_head, adapter, assembler), instrument_source_kernels(collector, carbon_kernels):
            loss, auxiliary = objective(jax.device_put(parameters), jax.device_put(sample))
            jax.block_until_ready((loss, auxiliary))
        report["scalar_forward_seconds_including_boundary_persistence"] = time.monotonic()-forward_started
        report["scalar_daily_loss"] = float(np.asarray(loss))
        report["actual_next_state_binding"] = capture_tree(collector, "actual_next_state_binding", auxiliary["state_diagnostics"])
        report["scalar_auxiliary"] = jax.device_get({key: value for key, value in auxiliary.items() if key != "state_diagnostics"})
        hard = owner.hard_counts(auxiliary)
        report["hard_counts"] = hard
        report["physical_hard_counts_zero"] = not any(hard.values())
        # Preserve the original auxiliary finite gate; the public hard counts
        # already apply the physical defined masks to the captured next state.
        report["forward_finite"] = bool(np.isfinite(report["scalar_daily_loss"]) and all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(report["scalar_auxiliary"])))
        if args.reference_state_diagnostics:
            previous = read_pickle(args.reference_state_diagnostics)
            matches = [item for item in previous["diagnostics"] if item["chain_step"] == args.reference_step and item["artifact_id"] == args.artifact_id]
            if len(matches) != 1: raise ValueError("saved daily reference step/artifact association differs")
            daily = matches[0][args.schedule]["daily"]
            report["saved_daily32_component"] = {"loss": daily["loss"], "auxiliary": daily["auxiliary"],
                "reference_step": args.reference_step, "artifact_id": args.artifact_id,
                "comparison_scope": "scalar row forward versus existing mean32; no scalar/mean parity claim"}
            collector.put("saved_actual_row_state_gradient_mean32", daily["initial_state_gradient"][args.row])
            collector.put("saved_actual_micro4_state_gradient_mean4", daily["initial_state_gradients_micro4_mean"][args.row//4])
        observed = {record["symbol"] for record in collector.records}
        report["boundary_coverage"] = {name: name in observed for name in KERNELS}
        report["status"] = ("captured" if any(report["boundary_coverage"].values()) else "forward_only_no_target_boundaries") if report["forward_finite"] and report["physical_hard_counts_zero"] else "forward_gate_failed"
        persist_capture(args.output, report, collector)
        if args.physical_state_vjp and report["status"] == "captured":
            report["physical_tail_state_AD"] = True
            emit("frozen_neural_physical_tail_state_vjp_entered")
            began = time.monotonic()
            device_parameters, device_sample = jax.device_put(parameters), jax.device_put(sample)
            def physical_loss(state):
                changed = device_sample._replace(inputs=device_sample.inputs._replace(continuous_state=state))
                return objective(device_parameters, changed)[0]
            # No physical instrumentation during AD: reverse tracers cannot be
            # represented as concrete actual boundary operands. The completed
            # forward capture above supplies those operands independently.
            repeats = []
            for repeat in range(args.physical_state_vjp_repeats):
                repeat_started = time.monotonic()
                with ExitStack() as derivative_patches:
                    derivative_patches.enter_context(jax.disable_jit())
                    derivative_patches.enter_context(owner.candidate_process_head_patches(shared_process_head, adapter, assembler))
                    if args.capture_turnover_cotangent:
                        def capture_cotangent(value):
                            report.setdefault("actual_turnover_output_cotangents", []).append({
                                "repeat": repeat+1,
                                "arrays": capture_tree(collector, f"actual_turnover_output_cotangent.repeat{repeat+1}", value),
                                "AD_boundary_note": "custom_vjp uses native original pullback; added boundary may reassociate floating-point reverse operations"})
                            emit("actual_turnover_output_cotangent_captured", repeat=repeat+1)
                        derivative_patches.enter_context(observe_turnover_output_cotangent(carbon_kernels, capture_cotangent))
                    frozen_loss, state_gradient = jax.value_and_grad(physical_loss)(device_sample.inputs.continuous_state)
                    jax.block_until_ready((frozen_loss, state_gradient))
                row = {"repeat": repeat+1, "seconds": time.monotonic()-repeat_started,
                    "loss": float(np.asarray(frozen_loss)),
                    "gradient": collector.put(f"frozen_neural_physical_tail_repeat{repeat+1}_state_gradient", state_gradient)}
                if repeats:
                    first = collector.arrays[repeats[0]["gradient"]["array"]]
                    row["same_first_gradient"] = bool(np.array_equal(first, np.asarray(state_gradient)))
                    row["same_first_loss"] = repeats[0]["loss"] == row["loss"]
                repeats.append(row)
                report["physical_tail_vjp_repeats"] = repeats
                emit("frozen_neural_physical_tail_state_vjp_repeat_returned", repeat=repeat+1)
            report["physical_tail_state_vjp"] = {
                "scope": "actual_scalar_physical_tail_only_neural_context_and_raw_outputs_held_constant",
                "omitted_path": "all neural output derivatives with respect to initial state; not full-network state AD parity",
                "parameter_AD": False, "scalar_loss": float(np.asarray(frozen_loss)),
                "finite_gradient": bool(np.all(np.isfinite(np.asarray(state_gradient)))),
                "same_forward_loss": bool(np.asarray(frozen_loss) == np.asarray(loss)),
                "gradient": collector.put("frozen_neural_physical_tail_scalar_state_gradient", state_gradient),
                "mean32_row_gradient": collector.put("frozen_neural_physical_tail_mean32_row_state_gradient", state_gradient/32),
                "coordinate2243_scalar_gradient": float(np.asarray(state_gradient)[2243]),
                "coordinate2243_mean32_row_gradient": float(np.asarray(state_gradient)[2243]/32),
                "seconds": time.monotonic()-began,
                "historical_binary_cause_established": False}
            emit("frozen_neural_physical_tail_state_vjp_returned")
        return report
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        persist_capture(args.output, report, collector)
        raise
    finally:
        source_patches.close()
        timer.cancel()


if __name__ == "__main__":
    import copy
    args = parser().parse_args()
    if args.additional_capture and (args.neural_cache or args.physical_state_vjp):
        raise ValueError("different-row batch cannot reuse a single sample's neural cache or physical VJP")
    plans = [(args.schedule, args.row), *(tuple(value.split(":")) for value in args.additional_capture)]
    if len(plans) > 4 or len(set((schedule, int(row)) for schedule, row in plans)) != len(plans):
        raise ValueError("at most four distinct actual schedule/row captures required")
    manifest = json.loads(args.manifest.read_text())
    statuses = []
    for index, (schedule, row) in enumerate(plans):
        current = copy.copy(args)
        if index:
            current.schedule, current.row = schedule, int(row)
            current.chain_snapshot = Path(manifest["resume"][schedule])
            current.output = args.output.with_name(args.output.stem+f".{schedule}-row{row}"+args.output.suffix)
        result = run(current)
        statuses.append(result["status"])
    raise SystemExit(0 if all(status == "captured" for status in statuses) else 1)
