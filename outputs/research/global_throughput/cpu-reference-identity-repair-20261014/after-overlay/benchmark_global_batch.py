"""Bounded B1/B4 complete-physics batch equivalence and timing diagnostic.

Writes one consolidated report and event log, preserving failure phases.
No scientific annual/Fortran gate is inferred from this 14 later-day window.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pickle
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESS_STARTED = float(os.environ.get("ORCHJAX_GLOBAL_BATCH_PROCESS_START", time.perf_counter()))
sys.path.insert(0, str(ROOT))

import jax
import jaxlib
import numpy as np

try:
    from jax._src.api_util import shaped_abstractify as _shaped_abstractify
except ImportError:  # JAX 0.10 moved this internal helper to core.
    from jax._src.core import shaped_abstractify as _shaped_abstractify

try:
    import psutil
except ModuleNotFoundError:
    psutil = None

from jax_orchidee.driver import global_batch as batch
from jax_orchidee.driver import orchestration as driver
from jax_orchidee.driver.fast_state import fast_state_from_previous_packet
from jax_orchidee.driver.reference_layout import resolve_paper_landpoint_reference
from jax_orchidee.driver.run_def_materialization import (
    materialize_case_run_def_values,
    read_run_def_values,
    write_materialized_run_def,
)
from scripts.dev.global_batch_stage_store import StageStore
from scripts.dev.probe_global_batch_cache import CacheEvents, atomic_bytes, atomic_json, configure_cache

POINTS = ("001.0-071.0", "069.0-119.0", "319.0-057.0", "331.0-057.0")
BASELINE = "133e35aec7b83b9d79888c46092f64d518e97299"


def rss_bytes():
    if psutil is not None:
        return psutil.Process().memory_info().rss
    if sys.platform.startswith("linux"):
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    return 0  # The metadata explicitly marks unavailable instrumentation.


def affinity():
    if psutil is not None:
        return psutil.Process().cpu_affinity()
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return None


def linux_ru_maxrss_bytes():
    if sys.platform.startswith("linux"):
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return None


def sync(value):
    for leaf in jax.tree_util.tree_leaves(value):
        ready = getattr(leaf, "block_until_ready", None)
        if ready is not None:
            ready()
    jax.effects_barrier()


def metrics(samples, point_days):
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "valid_point_days": point_days,
        "point_days_per_second": point_days / statistics.median(samples),
    }


def compare(actual, reference, *, sentinel_paths=None):
    """Fixed elementwise combined gate; schema/dtype/discrete exact.

    Only an explicitly identified roughheight_pft bare-soil coordinate may
    hold matched NaN sentinels. All other nonfinite values are failures.
    """
    ap, at = jax.tree_util.tree_flatten_with_path(actual)
    rp, rt = jax.tree_util.tree_flatten_with_path(reference)
    failures, max_abs, max_ratio = [], 0.0, 0.0
    allowed_sentinels = []
    if at != rt:
        return {"passed": False, "failures": [{"reason": "schema"}]}
    for (path, av), (_, rv) in zip(ap, rp, strict=True):
        name = jax.tree_util.keystr(path)
        a, r = np.asarray(av), np.asarray(rv)
        if a.shape != r.shape or a.dtype != r.dtype:
            failures.append(
                {
                    "path": name,
                    "reason": "shape_or_dtype",
                    "actual": [list(a.shape), str(a.dtype)],
                    "reference": [list(r.shape), str(r.dtype)],
                }
            )
        elif a.dtype.kind in "biu":
            if not np.array_equal(a, r):
                failures.append({"path": name, "reason": "discrete"})
        else:
            if not np.all(np.isfinite(a)) or not np.all(np.isfinite(r)):
                contract = (sentinel_paths or {}).get(name)
                allowed = np.zeros(a.shape, dtype=bool)
                if contract and a.ndim in (2, 3) and a.shape[-2:] == tuple(contract["inner_shape"]):
                    allowed[..., contract["bare_soil_index"]] = True
                matched_nan = np.isnan(a) & np.isnan(r) & allowed
                unexpected = ((~np.isfinite(a)) | (~np.isfinite(r))) & ~matched_nan
                if np.any(unexpected):
                    failures.append(
                        {
                            "path": name,
                            "reason": "nonfinite",
                            "actual_count": int(np.count_nonzero(~np.isfinite(a))),
                            "reference_count": int(np.count_nonzero(~np.isfinite(r))),
                            "unexpected_coordinates": np.argwhere(unexpected).tolist(),
                        }
                    )
                    continue
                count = int(np.count_nonzero(matched_nan))
                allowed_sentinels.append({"path": name, "count": count, "contract": contract})
                a = np.where(matched_nan, 0.0, a)
                r = np.where(matched_nan, 0.0, r)
            delta = np.abs(a - r)
            bound = 1e-8 + 1e-10 * np.abs(r)
            max_abs = max(max_abs, float(np.max(delta, initial=0)))
            ratio = float(np.max(delta / bound, initial=0))
            max_ratio = max(max_ratio, ratio)
            if ratio > 1:
                failures.append({"path": name, "reason": "tolerance", "max_bound_ratio": ratio})
    return {
        "passed": not failures,
        "compared_leaves": len(ap),
        "max_abs": max_abs,
        "max_bound_ratio": max_ratio,
        "failures": failures,
        "atol": 1e-8,
        "rtol": 1e-10,
        "nonfinite_policy": "fail_except_exact_source_backed_bare_soil_roughheight_nan_mask",
        "allowed_sentinels": allowed_sentinels,
    }


class Recorder:
    def __init__(self, output):
        self.output = output
        window = os.environ.get("ORCHJAX_GLOBAL_BATCH_WINDOW_ID")
        self.window_report = Path(window) / "benchmark-report.json" if window else None
        output.mkdir(parents=True, exist_ok=True)
        self.result = {"status": "running", "phases": [], "gates": [], "timing": {}}
        self.sentinel_paths = {}
        self.started = time.perf_counter()
        self.stop = threading.Event()
        self.peak = rss_bytes()
        self.sampler = threading.Thread(target=self.sample, daemon=True)
        self.sampler.start()

    def sample(self):
        while not self.stop.wait(0.1):
            self.peak = max(self.peak, rss_bytes())

    def flush(self):
        self.result["peak_rss_bytes"] = self.peak
        self.result["linux_ru_maxrss_peak_bytes"] = linux_ru_maxrss_bytes()
        self.result["elapsed_seconds"] = time.perf_counter() - self.started
        atomic_json(self.output / "report.json", self.result)
        if self.window_report is not None:
            atomic_json(self.window_report, self.result)

    def phase(self, name, **kwargs):
        entry = {
            "phase": name,
            "elapsed_seconds": time.perf_counter() - self.started,
            "rss_bytes": rss_bytes(),
            **kwargs,
        }
        self.result["phases"].append(entry)
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self.flush()
        print(json.dumps(entry), flush=True)

    def gate(self, name, actual, reference):
        entry = {"name": name, **compare(actual, reference, sentinel_paths=self.sentinel_paths)}
        self.result["gates"].append(entry)
        self.flush()
        if not entry["passed"]:
            raise AssertionError(f"gate failed: {name}; see report.json")


def metadata(args):
    cpu_affinity = affinity()
    return {
        "command": sys.argv,
        "cwd": str(ROOT),
        "platform": platform.platform(),
        "python": sys.version,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "devices": [str(d) for d in jax.devices()],
        "device_details": [{"platform": d.platform, "kind": d.device_kind, "id": d.id} for d in jax.devices()],
        "backend_platform_version": str(jax.devices()[0].client.platform_version),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": cpu_affinity,
        "cpu_model": platform.processor(),
        "hostname": platform.node(),
        "physical_gpu": {"uuid": os.environ.get("ORCHJAX_PHYSICAL_GPU_UUID"),
                         "driver_version": os.environ.get("ORCHJAX_GPU_DRIVER_VERSION"),
                         "source": "launcher-selected GPU; original numerical manifest lacks physical UUID/driver"},
        "loadavg": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "env": {
            k: os.environ.get(k)
            for k in (
                "XLA_FLAGS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "JAX_PLATFORMS",
                "JAX_COMPILATION_CACHE_DIR",
                "ORCHJAX_DISABLE_COMPILATION_CACHE",
                "CUDA_VISIBLE_DEVICES",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
            )
        },
        "cache_policy": "disabled"
        if os.environ.get("ORCHJAX_DISABLE_COMPILATION_CACHE") == "1"
        else "persistent_cache_may_hit",
        "gpu_count": sum(d.platform == "gpu" for d in jax.devices()),
        "visible_affinity_core_count": len(cpu_affinity) if cpu_affinity is not None else None,
        "assigned_cpu_count": os.environ.get("SLURM_CPUS_PER_TASK"),
        "rss_instrumentation": "psutil_current_RSS_100ms_sample"
        if psutil is not None
        else (
            "proc_statm_current_RSS_100ms_sample_plus_resource_ru_maxrss_Linux_KiB_converted_to_bytes"
            if sys.platform.startswith("linux")
            else "unavailable"
        ),
        "warm_reset_policy": "each repeat uses the same frozen Day1 end state; two sequential seven-day blocks",
        "block_days": [[2, 8], [9, 15]],
        "inner_npts": 1,
        "baseline_ref": args.baseline_ref,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "fortran_science_gate": "not_run; separate annual/source acceptance contract",
    }


def legacy_factory(ref):
    """Load only the old factory from the pinned Git source for refactor gate."""
    source = subprocess.check_output(
        ["git", "show", f"{ref}:jax_orchidee/driver/orchestration.py"], cwd=ROOT, text=True
    )
    tree = ast.parse(source)
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_paper_compiled_later_day_block_executable"
    )
    namespace = vars(driver).copy()
    namespace["_COMPILED_LATER_DAY_BLOCK_CACHE"] = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), f"git:{ref}", "exec"), namespace)
    original_transition = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "transition")
    candidate_tree = ast.parse((ROOT / "jax_orchidee/driver/orchestration.py").read_text(encoding="utf-8"))
    factory = next(
        n
        for n in candidate_tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_paper_later_day_block_transition_factory"
    )
    candidate_transition = next(n for n in factory.body if isinstance(n, ast.FunctionDef) and n.name == "transition")
    if ast.dump(original_transition, include_attributes=False) != ast.dump(
        candidate_transition, include_attributes=False
    ):
        raise AssertionError("extracted transition AST differs from pinned original")
    return namespace[node.name]


def input_records(args):
    if args.input_manifest:
        return json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if args.asset_root:
        return [
            {
                "landpoint_id": p,
                "run_def": str(args.asset_root / "landpoints" / p / "runtime_used_run.def"),
                "reference_dir": str(args.asset_root / "landpoints" / p / "reference"),
            }
            for p in POINTS
        ]
    result = []
    for p in POINTS:
        r = resolve_paper_landpoint_reference(args.truth_root, p)
        if r.output_dir is None or r.run_def is None:
            raise FileNotFoundError(f"independent reference package missing: {p}")
        values = materialize_case_run_def_values(
            base_used_run_def=r.used_run_def
            or args.truth_root / "outputs/server_1961_trace_full_20260623/run/used_run.def",
            case_run_def=r.run_def,
            base_is_fortran_used_truth=r.used_run_def is not None,
        )
        dest = args.output_dir / "run_defs" / p / "used_run.def"
        write_materialized_run_def(values, dest)
        result.append({"landpoint_id": p, "run_def": str(dest), "reference_dir": str(r.output_dir)})
    return result


def prepare(args, recorder):
    if args.prepared_input:
        with args.prepared_input.open("rb") as f:
            lanes, second = pickle.load(f)
        from dataclasses import replace

        for lane in lanes:
            if read_run_def_values(lane.context.run_def_path) != lane.context.run_def_values:
                raise ValueError("prepared run.def changed since preparation")
            computed = batch.static_signature(lane.context, lane.initial_state, lane.tables, lane.dispatch)
            if args.mode == "staged" and computed != lane.static_signature:
                raise ValueError("stored prepared static signature changed; staged admission refuses refresh")
        lanes = [
            replace(
                lane,
                static_signature=batch.static_signature(lane.context, lane.initial_state, lane.tables, lane.dispatch),
            )
            for lane in lanes
        ]
        recorder.phase("prepared_input_reused", path=str(args.prepared_input))
        return lanes, second
    records = input_records(args)
    recorder.result["input_records"] = records
    lanes, second = [], []
    cold_checkpoint = args.output_dir / "cold_day1_states.pkl"
    if cold_checkpoint.exists():
        with cold_checkpoint.open("rb") as f:
            cold_states = pickle.load(f)
    else:
        cold_states = {}
    for record in records:
        point = record["landpoint_id"]
        # Preserve immutable assets; normalize only the declared data-file path
        # in a new materialized run.def. Scientific parameter strings stay exact.
        values = read_run_def_values(record["run_def"])
        if "VEGETATION_FILE" in values:
            values["VEGETATION_FILE"] = str(
                Path(os.environ["ORCHIDEE_DATA_ROOT"]) / "INPUTDIR_ZZ/PFT1860_mangr_025deg.nc"
            ).replace("\\", "/")
        run_def = args.output_dir / "run_defs" / point / "used_run.def"
        write_materialized_run_def(values, run_def)
        recorder.phase("prepare_context", landpoint_id=point)
        context = driver.prepare_paper_1961_driver_context(
            args.config, used_run_def_path=run_def, reference_run_dir=record["reference_dir"]
        )
        cold_cache = args.output_dir / "run_defs" / point / "cold_day1.pkl"
        recorder.phase("cold_day1_start", landpoint_id=point)
        identity = (values, args.config.read_bytes(), str(record["reference_dir"]))
        if point in cold_states:
            signature, cold_state = cold_states[point]
            if signature != identity:
                raise ValueError("cold cache scientific input identity changed")
            recorder.phase("cold_day1_reused", landpoint_id=point)
        elif cold_cache.exists():
            with cold_cache.open("rb") as f:
                signature, cold_state = pickle.load(f)
            if signature != identity:
                raise ValueError("cold cache scientific input identity changed")
            recorder.phase("cold_day1_reused", landpoint_id=point)
        else:
            first = driver.paper_1961_driver_cold_start_day_scaffold(
                args.config,
                year=1961,
                used_run_def_path=run_def,
                reference_run_dir=record["reference_dir"],
                prepared_context=context,
            )
            if not first.ready_for_first_day_end_state or first.first_day_end_state is None:
                raise RuntimeError(f"cold Day1 incomplete: {point}: {first.missing_components}")
            cold_state = first.first_day_end_state
            sync(cold_state.fields_by_component)
        cold_states[point] = (identity, cold_state)
        with cold_checkpoint.open("wb") as f:
            pickle.dump(cold_states, f, protocol=pickle.HIGHEST_PROTOCOL)
        lane = batch.prepare_lane(point, context, cold_state)
        # Prepare second forcing block independently, using an aligned copy only
        # for the host adapter. Execution takes the actual first block end state.
        shifted = driver.DriverPreviousStepStatePacket(
            tstep=8 * 48 - 1,
            fields_by_component=lane.initial_state.fields_by_component,
            provenance_by_component=lane.initial_state.provenance_by_component,
        )
        second_lane = batch.prepare_lane(point, context, shifted, day_numbers=range(9, 16))
        lanes.append(lane)
        second.append(second_lane.arguments)
        recorder.phase(
            "lane_prepared",
            landpoint_id=point,
            ready=True,
            missing=[],
            actual_path="cold_day1_then_compiled_complete_day_blocks",
        )
    destination = args.output_dir / "prepared_inputs.pkl"
    with destination.open("wb") as f:
        pickle.dump((lanes, second), f, protocol=pickle.HIGHEST_PROTOCOL)
    recorder.phase("prepared_inputs_saved", path=str(destination))
    return lanes, second


def _signature_leaf(value):
    aval = _shaped_abstractify(value)
    return tuple(aval.shape), str(aval.dtype), bool(getattr(aval, "weak_type", False))


def _argument_signature(arguments):
    leaves, tree = jax.tree_util.tree_flatten((tuple(arguments), {}))
    return tree, tuple(_signature_leaf(value) for value in leaves)


def _aval_signature(avals):
    return tuple(_signature_leaf(value) for value in avals)


def describe_aot_signature(stage, arguments, *, lowered=False):
    """Preserve failed admission metadata without changing JAX internals."""
    pairs, tree = jax.tree_util.tree_flatten_with_path((tuple(arguments), {}))
    expected = _aval_signature([value for _, value in pairs])
    public = _aval_signature(jax.tree_util.tree_leaves(stage.in_avals))
    mesh = stage._lowering if lowered else stage._executable
    info = mesh.compile_args.get("all_args_info") if lowered else getattr(mesh, "_all_args_info", None)
    kept = mesh.compile_args.get("kept_var_idx") if lowered else getattr(mesh, "_kept_var_idx", None)
    internal = None if info is None else _aval_signature(info.in_avals)
    return {"caller_leaves": len(expected), "public_input_leaves": len(public),
            "internal_all_args_leaves": None if internal is None else len(internal),
            "kept_input_leaves": None if kept is None else len(kept),
            "public_tree_shape_dtype_weak_type_equal": stage.in_tree == tree and public == expected,
            "internal_signature_equal": None if internal is None else internal == expected,
            "caller_paths": [str(path) for path, _ in pairs], "caller_signature": expected,
            "public_signature": public, "internal_signature": internal,
            "stage_id": id(stage), "lowering_or_executable_id": id(mesh), "all_args_info_id": id(info),
            "kept_indices": None if kept is None else sorted(kept),
            "debug_info": None if info is None else str(getattr(info, "debug_info", None))}


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


def run_two(executable, first, second):
    a = executable(*first)
    b = executable(*second._replace(values_by_component=a[0]))
    sync((a, b))
    return a, b


def lower_compile(fn, arguments, name, recorder):
    recorder.phase(name + "_lower_start")
    t = time.perf_counter()
    lowered = jax.jit(fn).lower(*arguments)
    recorder.result["timing"][name + "_lower_seconds"] = time.perf_counter() - t
    compilation = {"lowered_input_metadata": describe_aot_signature(lowered, arguments, lowered=True)}
    recorder.result.setdefault("compilations", {})[name] = compilation
    try:
        lowered_signature = audit_aot_signature(lowered, arguments, lowered=True)
    except TypeError as error:
        compilation["signature_admission_error"] = str(error)
        recorder.phase(name + "_lower_signature_rejected")
        raise
    ir = str(lowered.compiler_ir(dialect="stablehlo"))
    compilation.update(stablehlo_sha256=hashlib.sha256(ir.encode()).hexdigest(),
                       lowered_input_signature=lowered_signature)
    recorder.phase(name + "_compile_start")
    t = time.perf_counter()
    observer = getattr(recorder, "cache_observer", None)
    event_start = len(observer.events) if observer is not None else 0
    executable = lowered.compile()
    recorder.result["timing"][name + "_compile_seconds"] = time.perf_counter() - t
    compilation["compiled_input_signature"] = audit_aot_signature(executable, arguments)
    compilation["call_binding"] = "explicit stage Compiled.call parameters; standard JAX checks retained"
    compilation["cache_events"] = observer.events[event_start:] if observer is not None else []
    recorder.phase(name + "_compiled")
    return BoundAOTExecutable(executable)


def independent_scalar_executable(lane, name, recorder):
    """Prevent old nested structural cache from binding another point's closure."""
    before = len(driver._COMPILED_SECHIBA_SCAN_CACHE)
    driver._COMPILED_SECHIBA_SCAN_CACHE.clear()
    recorder.phase(
        "independent_scalar_nested_cache_cleared",
        landpoint_id=lane.landpoint_id,
        entries_before=before,
        entries_after=0,
    )
    return lower_compile(batch.make_scalar_transition(lane), lane.arguments, name, recorder)


def stack_second(lanes, arguments):
    from dataclasses import replace

    return batch.stack_lanes([replace(lane, arguments=a) for lane, a in zip(lanes, arguments, strict=True)])


def roughheight_sentinel_contract(lane):
    scalars = lane.context.run_scalars
    if scalars.pft_ids[0] != "bare_soil" or int(scalars.fortran_pft_ids[0]) != 1:
        raise ValueError("roughheight sentinel requires stable bare-soil PFT identity at index zero")
    spec = fast_state_from_previous_packet(lane.initial_state).spec
    paths = {}
    for component in ("diffuco_previous_step_state", "sechiba_finalize_state"):
        ci = spec.components.index(component)
        fi = spec.field_names_by_component[ci].index("roughheight_pft")
        for block in (0, 1):
            paths[f"[{block}][0][{ci}][{fi}]"] = {
                "component": component,
                "field": "roughheight_pft",
                "bare_soil_index": 0,
                "inner_shape": [1, scalars.nvm],
                "provenance": "condveg.f90::condveg_z0cdrag 1325-1474 / condveg_z0cdrag_dyn 1578-1718; JAX condveg.py NaN initialization and vegetation loop starting at index1; tests/unit/test_condveg.py bare-soil NaN assertions; diffuco.py:4166-4169 masks bare-soil roughheight before arithmetic",
            }
    return paths


def save_reference(path, identity, value):
    sync(value)
    atomic_bytes(path, pickle.dumps((identity, jax.device_get(value)), protocol=pickle.HIGHEST_PROTOCOL))


def reference_identity(args, lane, second):
    # Small bounded prepared inputs, not a bulk dataset hash. A changed state,
    # forcing, parameter, template, closure or pinned factory invalidates reuse.
    return hashlib.sha256(
        pickle.dumps(
            (args.baseline_ref, lane.static_signature, lane.arguments, second), protocol=pickle.HIGHEST_PROTOCOL
        )
    ).hexdigest()


def benchmark(args, recorder, lanes, second):
    old_factory = legacy_factory(args.baseline_ref)
    first = lanes[0]
    recorder.sentinel_paths = roughheight_sentinel_contract(first)
    recorder.result["allowed_sentinel_contract"] = recorder.sentinel_paths
    legacy_path = args.output_dir / "legacy_original_reference.pkl"
    identity = reference_identity(args, first, second[0])
    if legacy_path.exists():
        with legacy_path.open("rb") as f:
            saved_identity, old_results = pickle.load(f)
        if identity != saved_identity:
            raise ValueError("legacy reference inputs changed; select a new output directory")
        recorder.phase("legacy_original_reference_reused", path=str(legacy_path))
    else:
        recorder.phase("legacy_aot_compile_start")
        t = time.perf_counter()
        old, _ = old_factory(
            args.config,
            context=first.context,
            initial_state=first.initial_state,
            year=1961,
            block_forcing=first.arguments.forcing_days,
            block_day_numbers=first.arguments.science_day_numbers,
            prebound_hydrol_runtime_static_tables=first.tables,
            daily_carbon_dispatch=first.dispatch,
        )
        recorder.result["timing"]["legacy_aot_compile_seconds"] = time.perf_counter() - t
        t = time.perf_counter()
        old_results = run_two(old, first.arguments, second[0])
        recorder.result["timing"]["legacy_first_two_block_call_seconds"] = time.perf_counter() - t
        save_reference(legacy_path, identity, old_results)
        recorder.phase("legacy_original_reference_saved", path=str(legacy_path))
    scalar = independent_scalar_executable(first, "extracted_scalar", recorder)
    extracted_results = run_two(scalar, first.arguments, second[0])
    save_reference(args.output_dir / "extracted_scalar_reference.pkl", identity, extracted_results)
    recorder.gate("B1_original_AOT_vs_extracted_full_state_day_outputs", extracted_results, old_results)
    # Independent per-point closures are a ground truth for the compatibility
    # audit itself, including any accidentally omitted context numeric value.
    scalars = []
    scalar_device = []
    references = []
    if args.resume_references:
        origin = args.reference_origin_report
        cleared = {
            p.get("landpoint_id")
            for p in origin["phases"]
            if p["phase"] == "independent_scalar_nested_cache_cleared" and p.get("entries_after") == 0
        }
        if not {lane.landpoint_id for lane in lanes}.issubset(cleared):
            raise ValueError("resume requires recorded independent per-point nested-cache clearing")
        scientific_diff = subprocess.check_output(
            ["git", "diff", "--name-only", "a33b82da9a1233f38e00d09435aecde4ba088111", "--", "jax_orchidee"],
            cwd=ROOT,
            text=True,
        )
        if scientific_diff.strip():
            raise ValueError("scientific source changed since cached independent references")
        recorder.result["reference_origin"] = {
            "report": str(args.output_dir / "reference_origin_report.json"),
            "commit": origin["environment"]["git_head"],
            "policy": "reuse completed cache-independent references; verify pooled compatible scalar against each before timing",
        }
    for index, (lane, s) in enumerate(zip(lanes, second, strict=True)):
        fn = (
            scalar
            if index == 0 or args.resume_references
            else independent_scalar_executable(lane, "scalar_" + lane.landpoint_id, recorder)
        )
        scalars.append(fn)
        t = time.perf_counter()
        device_pair = jax.device_put((lane.arguments, s))
        sync(device_pair)
        scalar_device.append(device_pair)
        recorder.result["timing"]["scalar_" + lane.landpoint_id + "_device_transfer_seconds"] = time.perf_counter() - t
        reference_path = args.output_dir / ("scalar_" + lane.landpoint_id + "_reference.pkl")
        if args.resume_references:
            with reference_path.open("rb") as f:
                saved_identity, reference = pickle.load(f)
            if saved_identity != reference_identity(args, lane, s):
                raise ValueError(f"independent cached reference inputs changed: {lane.landpoint_id}")
            recorder.gate(
                "pooled_scalar_vs_cached_independent_" + lane.landpoint_id, run_two(fn, *device_pair), reference
            )
            recorder.phase(
                "independent_scalar_reference_reused", landpoint_id=lane.landpoint_id, path=str(reference_path)
            )
        else:
            reference = run_two(fn, *device_pair)
        references.append(reference)
        if not args.resume_references:
            save_reference(reference_path, reference_identity(args, lane, s), reference)
    recorder.phase("independent_scalar_references_complete")
    for size in args.sizes:
        group, other = lanes[:size], second[:size]
        inputs, second_inputs = batch.stack_lanes(group), stack_second(group, other)
        t = time.perf_counter()
        device_inputs, device_second = jax.device_put((inputs, second_inputs))
        sync((device_inputs, device_second))
        recorder.result["timing"][f"B{size}_device_transfer_seconds"] = time.perf_counter() - t
        executable = lower_compile(batch.make_batch_transition(group), device_inputs, f"B{size}", recorder)
        t = time.perf_counter()
        actual = run_two(executable, device_inputs, device_second)
        recorder.result["timing"][f"B{size}_first_call_seconds"] = time.perf_counter() - t
        save_reference(args.output_dir / f"B{size}_candidate_before_gate.pkl", identity, actual)
        for i, ref in enumerate(references[:size]):
            recorder.gate(
                f"B{size}_lane_{i}_both_blocks_full_state_day_outputs",
                jax.tree_util.tree_map(lambda v: v[i], actual),
                ref,
            )
        if size > 1:
            order = list(reversed(range(size)))
            perm_inputs = batch.stack_lanes([group[i] for i in order])
            perm_second = stack_second([group[i] for i in order], [other[i] for i in order])
            perm_result = run_two(executable, perm_inputs, perm_second)
            recorder.gate(
                f"B{size}_permutation", perm_result, jax.tree_util.tree_map(lambda v: v[np.asarray(order)], actual)
            )
            # Replace one lane with a different real point's complete input
            # bundle. All other state/forcing/templates/params remain frozen.
            replaced_group = group[:-1] + [group[0]]
            replaced_other = other[:-1] + [other[0]]
            isolated = run_two(
                executable, batch.stack_lanes(replaced_group), stack_second(replaced_group, replaced_other)
            )
            for i in range(size - 1):
                recorder.gate(
                    f"B{size}_replacement_unaffected_lane_{i}",
                    jax.tree_util.tree_map(lambda v: v[i], isolated),
                    references[i],
                )
            recorder.gate(
                f"B{size}_replacement_changed_lane", jax.tree_util.tree_map(lambda v: v[-1], isolated), references[0]
            )
        recorder.phase(f"B{size}_correctness_passed")
        warm, serial, wrapper = [], [], []
        for _ in range(args.warm_repeats):
            t = time.perf_counter()
            run_two(executable, device_inputs, device_second)
            warm.append(time.perf_counter() - t)
            t = time.perf_counter()
            for fn, device_pair in zip(scalars[:size], scalar_device[:size], strict=True):
                run_two(fn, *device_pair)
            serial.append(time.perf_counter() - t)
            t = time.perf_counter()
            wrapped = run_two(executable, batch.stack_lanes(group), stack_second(group, other))
            host = jax.device_get(wrapped)
            wrapper.append(time.perf_counter() - t)
        recorder.result["timing"][f"B{size}_warm_frozen_state_core"] = metrics(warm, size * 14)
        recorder.result["timing"][f"B{size}_scalar_serial"] = metrics(serial, size * 14)
        recorder.result["timing"][f"B{size}_preprepared_wrapper_stack_transfer_materialize"] = metrics(
            wrapper, size * 14
        )
        recorder.result["timing"][f"B{size}_core_speedup_vs_scalar_serial"] = statistics.median(
            serial
        ) / statistics.median(warm)
        t = time.perf_counter()
        host = jax.device_get(actual)
        recorder.result["timing"][f"B{size}_host_materialization_seconds"] = time.perf_counter() - t
        t = time.perf_counter()
        with (args.output_dir / f"B{size}_full_state_outputs.pkl").open("wb") as f:
            pickle.dump(host, f, protocol=pickle.HIGHEST_PROTOCOL)
        recorder.result["timing"][f"B{size}_full_result_io_seconds"] = time.perf_counter() - t
        recorder.phase(f"B{size}_warm_measurements_complete")


def scientific_source_digest():
    from scripts.dev.global_source_compatibility import source_identity

    return source_identity(ROOT)["science_sha256"]


def semantic_reference_identity(args, lane, second):
    """Content identity independent of pickle memoization and array placement."""
    digest = hashlib.sha256()
    digest.update(json.dumps({"baseline_ref": args.baseline_ref, "landpoint_id": lane.landpoint_id,
                              "static_signature": lane.static_signature}, sort_keys=True,
                             separators=(",", ":")).encode())
    values, structure = jax.tree_util.tree_flatten_with_path((lane.arguments, second))
    digest.update(str(structure).encode())
    for path, value in values:
        array = np.asarray(value)
        header = (jax.tree_util.keystr(path), array.shape, array.dtype.str)
        digest.update(json.dumps(header, separators=(",", ":")).encode() + b"\0")
        digest.update(array.tobytes(order="C") + b"\0")
    return digest.hexdigest()


def stage_identity(args, lanes, second, environment, *, include_git_head=False):
    """Project runtime identity; server consumers additionally bind execution HEAD.

    Generic legacy stage contracts retain their existing runtime schema.
    Server reference preparation opts in using the actual metadata producer.
    """
    runtime_keys = ("python", "jax", "jaxlib", "numpy", "device_details", "x64_enabled", "backend_platform_version")
    if include_git_head:
        runtime_keys += ("git_head",)
    compiler_keys = ("XLA_FLAGS", "JAX_PLATFORMS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    return {
        "contract": "four-real-points-two-seven-day-blocks-full-FP64-v1",
        "baseline_ref": args.baseline_ref,
        "science_sha256": scientific_source_digest(),
        "lanes": [{"landpoint_id": lane.landpoint_id, "inputs_sha256": semantic_reference_identity(args, lane, other)}
                  for lane, other in zip(lanes, second, strict=True)],
        "runtime": {key: environment[key] for key in runtime_keys},
        "compiler_environment": {key: environment["env"].get(key) for key in compiler_keys},
        "warm_repeats": args.warm_repeats,
        "gate": {"atol": 1e-8, "rtol": 1e-10, "discrete_schema_dtype": "exact",
                 "sentinel": "declared_bare_soil_roughheight_only"},
    }


def validate_legacy_origin(origin, environment, *, scientific_diff):
    """Check every recorded critical field; do not invent historical metadata."""
    previous = origin["environment"]
    for key in ("python", "jax", "jaxlib", "numpy", "device_details", "x64_enabled", "baseline_ref"):
        if key not in previous or previous[key] != environment[key]:
            raise ValueError(f"original GPU reference runtime/baseline mismatch: {key}")
    for key in ("XLA_FLAGS", "JAX_PLATFORMS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if key not in previous["env"] or previous["env"][key] != environment["env"].get(key):
            raise ValueError(f"original GPU reference compiler environment mismatch: {key}")
    if scientific_diff.strip():
        raise ValueError("original reference scientific source differs from current checkout")
    if not any(p["phase"] == "legacy_original_reference_saved" for p in origin["phases"]):
        raise ValueError("origin does not record a completed original scalar reference")
    return {
        "origin_commit": previous["git_head"],
        "checked": "all recorded Python/JAX/jaxlib/NumPy/FP64/device/compiler fields; full inputs; source Git diff",
        "missing_origin_metadata": ["backend_platform_version", "CUDA_driver_version", "source_content_hash_at_execution"],
        "limitation": "recorded origin commit/source correspondence; no claim that missing runtime fields were captured",
    }


def import_legacy_reference(args, recorder, lane, second):
    if args.legacy_reference_dir is None:
        raise ValueError("reference stages require --legacy-reference-dir with the completed original GPU reference")
    directory = args.legacy_reference_dir.resolve()
    origin = json.loads((directory / "report.json").read_text(encoding="utf-8-sig"))
    origin_commit = origin["environment"]["git_head"]
    scientific_diff = subprocess.check_output(
        ["git", "diff", "--name-only", origin_commit, "--", "jax_orchidee"], cwd=ROOT, text=True
    )
    evidence = validate_legacy_origin(origin, recorder.result["environment"], scientific_diff=scientific_diff)
    path = directory / "legacy_original_reference.pkl"
    data = path.read_bytes()
    saved_identity, value = pickle.loads(data)
    if saved_identity != reference_identity(args, lane, second):
        raise ValueError("original GPU reference full input identity mismatch")
    recorder.gate("legacy_original_reference_source_backed_nonfinite_policy", value, value)
    evidence.update(path=str(path), sha256=hashlib.sha256(data).hexdigest(), landpoint_id=lane.landpoint_id,
                    legacy_pickle_identity=saved_identity,
                    canonical_input_identity=semantic_reference_identity(args, lane, second),
                    role="original AOT lane001 reference only; not four independent extracted references")
    return value, evidence


def record_timing_origin(result, name, record, *, reused):
    for key in record.get("timing", {}):
        result.setdefault("timing_origins", {})[key] = {
            "stage": name, "reused_measurement": reused,
            "execution_origin": record.get("execution_origin", {"available": False, "reason": "origin not recorded"}),
        }


def staged_benchmark(args, recorder, lanes, second):
    """Candidate first, independently bound references next, validation/timing last."""
    if tuple(lane.landpoint_id for lane in lanes) != POINTS:
        raise ValueError("staged diagnostic requires the declared four real points in declared order")
    source = getattr(args, "stage_source_dir", None)
    if source is not None:
        StageStore.validate_source_destination(source, args.output_dir)
    legacy_factory(args.baseline_ref)  # Validate unchanged extracted transition AST before any reuse.
    recorder.sentinel_paths = roughheight_sentinel_contract(lanes[0])
    recorder.result["allowed_sentinel_contract"] = recorder.sentinel_paths
    store = StageStore(args.output_dir, stage_identity(args, lanes, second, recorder.result["environment"]))
    recorder.result["stages"] = store.manifest["stages"]
    required = ["B4_candidate", "legacy_original"] + ["scalar_" + lane.landpoint_id for lane in lanes]
    if source is not None:
        store.import_completed(source, required)
        recorder.phase("complete_numerical_prerequisites_imported", source=str(source), names=list(required),
                       policy="read-only source; original execution origins retained; no old measurement stages imported")
    b1_mode = getattr(args, "b1_mode", "vmapped")
    b1_stage = "B1_scalar_baseline" if b1_mode == "scalar" else "B1_validation"
    selected_point = getattr(args, "pooled_scalar_point", POINTS[0])
    selected_lane = next(lane for lane in lanes if lane.landpoint_id == selected_point)
    suffix = "" if selected_point == POINTS[0] else "_pool_" + selected_point
    batch_stage, serial_stage, b1_stage = "B4_batch_only_v1", "B4_scalar_comparison_v1" + suffix, b1_stage + suffix
    required += [batch_stage, serial_stage, b1_stage]
    recorder.result["b1_mode"] = b1_mode
    recorder.result["pooled_scalar"] = {
        "landpoint_id": selected_point, "context": "selected point's own independently audited context",
        "policy": "every new actual executable must pass all four independent reference gates before timing",
        "cache_hit": "not inferred from selection; inspect actual compile cache events",
    }
    recorder.result["required_stages"] = required
    for name in required:
        loaded = store.load(name)
        if loaded is not None:
            recorder.result["gates"].extend(loaded[1].get("gates", []))
            recorder.result["timing"].update(loaded[1].get("timing", {}))
            record_timing_origin(recorder.result, name, loaded[1], reused=True)
    recorder.phase("stage_identity_validated", manifest=str(store.path))

    def run_stage(name, operation):
        def execute():
            gate_start, phase_start = len(recorder.result["gates"]), len(recorder.result["phases"])
            old_timing = dict(recorder.result["timing"])
            old_compilations = dict(recorder.result.get("compilations", {}))
            started = time.perf_counter()
            recorder.phase(name + "_stage_start")
            value, provenance = operation()
            return value, {
                "provenance": provenance,
                "execution_origin": {
                    "available": True, "environment": recorder.result["environment"],
                    "report": str(recorder.window_report) if recorder.window_report is not None else None,
                    "report_limitation": None if recorder.window_report is not None else "window ID not supplied; shared latest report is mutable",
                    "pid": os.getpid(),
                    "process_start_monotonic": PROCESS_STARTED,
                    "pooled_scalar_point": selected_point if name in (serial_stage, b1_stage) else None,
                    "window_identity": os.environ.get("ORCHJAX_GLOBAL_BATCH_WINDOW_ID", "not supplied; use PID/start and supervisor context"),
                },
                "stage_seconds": time.perf_counter() - started,
                "gates": recorder.result["gates"][gate_start:],
                "phases": recorder.result["phases"][phase_start:],
                "timing": {k: v for k, v in recorder.result["timing"].items() if old_timing.get(k) != v},
                "compilations": {k: v for k, v in recorder.result.get("compilations", {}).items()
                                 if old_compilations.get(k) != v},
            }
        (value, record), reused = store.run(name, execute)
        record_timing_origin(recorder.result, name, record, reused=reused)
        recorder.phase(name + ("_stage_reused" if reused else "_stage_complete"))
        return value

    def batch_executable(size, name):
        # No preceding scalar closure may bind the batch's nested executable.
        driver._COMPILED_SECHIBA_SCAN_CACHE.clear()
        group = lanes[:size]
        started = time.perf_counter()
        host_inputs = batch.stack_lanes(group), stack_second(group, second[:size])
        recorder.result["timing"][name + "_host_stack_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        inputs = jax.device_put(host_inputs)
        sync(inputs)
        recorder.result["timing"][name + "_device_transfer_seconds"] = time.perf_counter() - started
        executable = lower_compile(batch.make_batch_transition(group), inputs[0], name, recorder)
        return executable, inputs

    def candidate():
        executable, inputs = batch_executable(4, "B4_candidate")
        started = time.perf_counter()
        actual = run_two(executable, *inputs)
        recorder.result["timing"]["B4_first_call_seconds"] = time.perf_counter() - started
        recorder.gate("B4_candidate_source_backed_nonfinite_policy", actual, actual)
        started = time.perf_counter()
        host = jax.device_get(actual)
        recorder.result["timing"]["B4_host_materialization_seconds"] = time.perf_counter() - started
        return host, {"role": "candidate actual compute; independent parity not yet certified", "inner_npts": 1}

    if args.stage in ("candidate", "all"):
        run_stage("B4_candidate", candidate)
    if args.stage in ("references", "all"):
        if store.load("B4_candidate") is None:
            raise ValueError("references require a complete saved B4 candidate stage")
        original = run_stage("legacy_original", lambda: import_legacy_reference(args, recorder, lanes[0], second[0]))
        for index, lane in enumerate(lanes):
            def reference(index=index, lane=lane):
                name = "scalar_" + lane.landpoint_id
                executable = independent_scalar_executable(lane, name, recorder)
                value = run_two(executable, lane.arguments, second[index])
                recorder.gate(name + "_source_backed_nonfinite_policy", value, value)
                if index == 0:
                    recorder.gate("B1_original_AOT_vs_extracted_full_state_day_outputs", value, original)
                return jax.device_get(value), {
                    "landpoint_id": lane.landpoint_id, "own_context_bound": True,
                    "nested_cache_cleared_before_lower": True,
                    "policy": "own lowered/compiled function and own inputs actually executed; disk reuse only by JAX key",
                }
            run_stage("scalar_" + lane.landpoint_id, reference)
    if args.stage in ("validate", "all"):
        missing = [name for name in required[:6] if store.load(name) is None]
        if missing:
            raise ValueError(f"validation requires independently complete prerequisites: {missing}")
        actual = store.load("B4_candidate")[0]
        original = store.load("legacy_original")[0]
        references = [store.load("scalar_" + lane.landpoint_id)[0] for lane in lanes]
        scalar = None
        batch_fn, batch_inputs = None, None

        def ensure_scalar():
            nonlocal scalar
            if scalar is None:
                scalar = independent_scalar_executable(selected_lane, "pooled_scalar_validation", recorder)
                for lane, other, ref in zip(lanes, second, references, strict=True):
                    recorder.gate("pooled_scalar_vs_independent_" + lane.landpoint_id,
                                  run_two(scalar, lane.arguments, other), ref)
                recorder.result["pooled_scalar"]["all_four_current_executable_gates_passed"] = True
            return scalar

        def ensure_batch():
            nonlocal batch_fn, batch_inputs
            if batch_fn is None:
                batch_fn, batch_inputs = batch_executable(4, "B4_validation")
            return batch_fn, batch_inputs

        def gates():
            executable, inputs = ensure_batch()
            started = time.perf_counter()
            current = run_two(executable, *inputs)
            recorder.result["timing"]["B4_validation_first_call_seconds"] = time.perf_counter() - started
            recorder.gate("B4_resumed_executable_vs_saved_candidate", current, actual)
            for index, (lane, other, ref) in enumerate(zip(lanes, second, references, strict=True)):
                recorder.gate(f"B4_lane_{index}_both_blocks_full_state_day_outputs",
                              jax.tree_util.tree_map(lambda v: v[index], current), ref)
            order = list(reversed(range(4)))
            permuted = run_two(executable, batch.stack_lanes([lanes[i] for i in order]),
                               stack_second([lanes[i] for i in order], [second[i] for i in order]))
            recorder.gate("B4_permutation", permuted, jax.tree_util.tree_map(lambda v: v[np.asarray(order)], actual))
            replaced = lanes[:-1] + [lanes[0]]
            isolated = run_two(executable, batch.stack_lanes(replaced), stack_second(replaced, second[:-1] + [second[0]]))
            for index in range(4):
                recorder.gate(f"B4_replacement_lane_{index}", jax.tree_util.tree_map(lambda v: v[index], isolated),
                              references[index if index < 3 else 0])
            return current

        def batch_only():
            # This complete snapshot must be published before any scalar compile/load.
            current = gates()
            executable, inputs = ensure_batch()
            warm, wrapper = [], []
            for _ in range(args.warm_repeats):
                started = time.perf_counter()
                run_two(executable, *inputs)
                warm.append(time.perf_counter() - started)
                started = time.perf_counter()
                wrapped = run_two(executable, batch.stack_lanes(lanes), stack_second(lanes, second))
                jax.device_get(wrapped)
                wrapper.append(time.perf_counter() - started)
            measured = {
                "B4_warm_frozen_state_core": metrics(warm, 56),
                "B4_preprepared_wrapper_stack_transfer_materialize": metrics(wrapper, 56),
            }
            recorder.result["timing"].update(measured)
            return {"outputs": jax.device_get(current), "measurements": measured}, {
                "path": "vmapped batch; no standalone scalar executable compiled before checkpoint publication",
                "independent_parity": True, "permutation": True, "replacement_isolation": True,
                "five_or_more_synchronized_repeats": True, "frozen_start": True,
                "accepted_scope": "B4 only; scalar comparison and B1 remain separate",
            }

        batch_saved = run_stage(batch_stage, batch_only)

        def measurements(size, executable, inputs, scalar_fn, *, scalar_baseline=False):
            label = "B1_scalar_baseline" if scalar_baseline else f"B{size}"
            started = time.perf_counter()
            device_pairs = jax.device_put([(lane.arguments, other) for lane, other in zip(lanes[:size], second[:size], strict=True)])
            sync(device_pairs)
            recorder.result["timing"][label + "_scalar_device_transfer_seconds"] = time.perf_counter() - started
            # A resumed timing stage also rechecks its actual scalar executable
            # against every previously independent reference before timing it.
            for index, pair in enumerate(device_pairs):
                recorder.gate(f"{label}_timing_scalar_lane_{index}", run_two(scalar_fn, *pair), references[index])
            warm, serial, wrapper = [], [], []
            for _ in range(args.warm_repeats):
                started = time.perf_counter()
                run_two(executable, *inputs)
                warm.append(time.perf_counter() - started)
                started = time.perf_counter()
                if not scalar_baseline:
                    for pair in device_pairs:
                        run_two(scalar_fn, *pair)
                    serial.append(time.perf_counter() - started)
                started = time.perf_counter()
                wrapped = (run_two(executable, lanes[0].arguments, second[0]) if scalar_baseline else
                           run_two(executable, batch.stack_lanes(lanes[:size]), stack_second(lanes[:size], second[:size])))
                jax.device_get(wrapped)
                wrapper.append(time.perf_counter() - started)
            result = {
                label + "_warm_frozen_state_core": metrics(warm, size * 14),
                label + ("_preprepared_wrapper_transfer_materialize" if scalar_baseline else
                         "_preprepared_wrapper_stack_transfer_materialize"): metrics(wrapper, size * 14),
            }
            if not scalar_baseline:
                result.update({label + "_scalar_serial": metrics(serial, size * 14),
                               label + "_core_speedup_vs_scalar_serial": statistics.median(serial) / statistics.median(warm)})
            recorder.result["timing"].update(result)
            return result, {"five_or_more_synchronized_repeats": True, "frozen_start": True,
                            "path": "single-point scalar program; no standalone vmapped B1 gate" if scalar_baseline else "vmapped batch",
                            "serial_policy": "pooled scalar verified against all independently executed lane references"}

        def scalar_comparison():
            executable = ensure_scalar()
            started = time.perf_counter()
            pairs = jax.device_put([(lane.arguments, other) for lane, other in zip(lanes, second, strict=True)])
            sync(pairs)
            recorder.result["timing"]["B4_scalar_device_transfer_seconds"] = time.perf_counter() - started
            samples = []
            for _ in range(args.warm_repeats):
                started = time.perf_counter()
                for pair in pairs:
                    run_two(executable, *pair)
                samples.append(time.perf_counter() - started)
            measured = {
                "B4_scalar_serial": metrics(samples, 56),
                "B4_core_speedup_vs_scalar_serial": statistics.median(samples) / statistics.median(
                    batch_saved["measurements"]["B4_warm_frozen_state_core"]["samples_seconds"]),
            }
            recorder.result["timing"].update(measured)
            return measured, {
                "serial_policy": "actual pooled scalar gated against all four independent references before timing",
                "comparison_batch_stage": batch_stage,
                "five_or_more_synchronized_repeats": True, "frozen_start": True,
            }

        run_stage(serial_stage, scalar_comparison)

        def b1():
            executable, inputs = batch_executable(1, "B1_validation")
            value = jax.tree_util.tree_map(lambda v: v[0], run_two(executable, *inputs))
            recorder.gate("B1_batch_vs_independent_scalar", value, references[0])
            recorder.gate("B1_batch_vs_original_AOT", value, original)
            return measurements(1, executable, inputs, ensure_scalar())

        def b1_scalar():
            executable = ensure_scalar()
            inputs = jax.device_put((lanes[0].arguments, second[0]))
            sync(inputs)
            value = run_two(executable, *inputs)
            recorder.gate("B1_scalar_baseline_vs_independent_scalar", value, references[0])
            recorder.gate("B1_scalar_baseline_vs_original_AOT", value, original)
            return measurements(1, executable, inputs, executable, scalar_baseline=True)

        run_stage(b1_stage, b1_scalar if b1_mode == "scalar" else b1)
    recorder.result["all_required_stages_complete"] = all(
        store.manifest["stages"].get(name, {}).get("status") == "complete" for name in required
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/orchidee_man_250919.yaml")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--truth-root", type=Path, default=ROOT)
    parser.add_argument("--prepared-input", type=Path)
    parser.add_argument(
        "--resume-references",
        action="store_true",
        help="Reuse this output directory's completed independent scalar references after an instrumentation-only failure.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 4], choices=[1, 4])
    parser.add_argument("--warm-repeats", type=int, default=5)
    parser.add_argument("--baseline-ref", default=BASELINE)
    parser.add_argument("--mode", choices=["audit", "benchmark", "staged"], default="benchmark")
    parser.add_argument("--stage", choices=["candidate", "references", "validate", "all"], default="all",
                        help="Staged mode: save actual B4 first, then each independent scalar, then correctness/timing/B1.")
    parser.add_argument("--legacy-reference-dir", type=Path,
                        help="Original scalar reference/report directory; all recorded runtime/source/input fields checked.")
    parser.add_argument("--stage-source-dir", type=Path,
                        help="Read-only staged source of six complete numerical prerequisites; fresh measurements remain separate.")
    parser.add_argument("--cache-mode", choices=["off", "persistent"], default="off")
    parser.add_argument("--cache-root", type=Path, help="Explicit persistent cache directory; no serialized AOT reuse.")
    parser.add_argument("--b1-mode", choices=["vmapped", "scalar"], default="vmapped",
                        help="Staged B1: standalone vmapped gate or explicitly labelled single-point scalar baseline.")
    parser.add_argument("--pooled-scalar-point", choices=POINTS, default=POINTS[0],
                        help="Staged serial context; actual executable must match all four independent references. Default unchanged.")
    args = parser.parse_args()
    if args.warm_repeats < 5:
        parser.error("at least five warm repeats required")
    if args.mode == "staged" and (args.prepared_input is None or args.resume_references):
        parser.error("staged mode requires --prepared-input and uses atomic stage recovery instead of --resume-references")
    if args.stage_source_dir is not None and (args.mode != "staged" or args.stage != "validate"):
        parser.error("--stage-source-dir requires --mode staged --stage validate")
    if args.cache_mode == "persistent" and args.cache_root is None:
        parser.error("persistent mode requires --cache-root")
    args.output_dir = args.output_dir.resolve()
    if args.stage_source_dir is not None:
        StageStore.validate_source_destination(args.stage_source_dir, args.output_dir)
    os.environ.setdefault("ORCHIDEE_REPO_ROOT", str(ROOT))
    os.environ.setdefault("ORCHIDEE_DATA_ROOT", str(args.truth_root / "data"))
    os.environ.setdefault("ORCHIDEE_REFERENCE_ROOT", str(args.truth_root / "reference"))
    os.environ["ORCHIDEE_OUTPUT_ROOT"] = str(args.output_dir)
    cache_policy = configure_cache(args.cache_mode, args.cache_root)
    if args.resume_references:
        origin_path = args.output_dir / "reference_origin_report.json"
        if not origin_path.exists():
            origin_path.write_bytes((args.output_dir / "report.json").read_bytes())
        args.reference_origin_report = json.loads(origin_path.read_text(encoding="utf-8-sig"))
    if args.mode == "staged" and (args.output_dir / "report.json").exists():
        atomic_bytes(args.output_dir / "windows" / f"prior_report_{time.time_ns()}.json",
                     (args.output_dir / "report.json").read_bytes())
    recorder = Recorder(args.output_dir)
    recorder.cache_observer = CacheEvents()
    previous_signals = {}

    def stop_requested(number, frame):
        raise InterruptedError(f"signal {number}; completed atomic stage checkpoints retained")

    if args.mode == "staged":
        for number in (signal.SIGTERM, signal.SIGINT):
            previous_signals[number] = signal.signal(number, stop_requested)
    try:
        recorder.result["environment"] = metadata(args)
        recorder.result["environment"]["explicit_cache_policy"] = cache_policy
        recorder.result["timing"]["import_setup_seconds"] = time.perf_counter() - PROCESS_STARTED
        if not jax.config.jax_enable_x64:
            raise RuntimeError("FP64 is required")
        recorder.phase("input_preparation_start")
        t = time.perf_counter()
        lanes, second = prepare(args, recorder)
        recorder.result["timing"]["input_preparation_seconds"] = time.perf_counter() - t
        recorder.result["compatibility_audit"] = batch.audit_compatible_lanes(lanes)
        recorder.phase("scientific_static_and_dynamic_schema_audit_passed")
        if args.mode == "benchmark":
            benchmark(args, recorder, lanes, second)
        elif args.mode == "staged":
            staged_benchmark(args, recorder, lanes, second)
        recorder.result["status"] = (
            "passed_audit_only" if args.mode == "audit" else
            "completed_stage_selection" if args.mode == "staged" and not recorder.result["all_required_stages_complete"] else
            "passed_bounded_forward_diagnostic"
        )
    except (InterruptedError, KeyboardInterrupt) as e:
        recorder.result.update(status="stopped", exception_type=type(e).__name__, message=str(e))
        recorder.phase("stopped", message=str(e))
        return 2
    except Exception as e:
        recorder.result.update(
            status="failed", exception_type=type(e).__name__, message=str(e), traceback=traceback.format_exc()
        )
        recorder.phase("failed", exception_type=type(e).__name__, message=str(e))
        return 1
    finally:
        recorder.stop.set()
        recorder.result["process_elapsed_seconds_including_import_setup"] = time.perf_counter() - PROCESS_STARTED
        recorder.flush()
        recorder.cache_observer.close()
        for number, previous in previous_signals.items():
            signal.signal(number, previous)
    recorder.phase("complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
