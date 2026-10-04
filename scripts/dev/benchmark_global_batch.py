"""Bounded B1/B4 complete-physics batch equivalence and timing diagnostic.

Writes one consolidated report and event log, preserving failure phases.
No scientific annual/Fortran gate is inferred from this 14 later-day window.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jaxlib
import numpy as np
import psutil

from jax_orchidee.driver import global_batch as batch
from jax_orchidee.driver import orchestration as driver
from jax_orchidee.driver.reference_layout import resolve_paper_landpoint_reference
from jax_orchidee.driver.run_def_materialization import (
    materialize_case_run_def_values,
    read_run_def_values,
    write_materialized_run_def,
)
from jax_orchidee.runtime import configure_jax_compilation_cache

POINTS = ("001.0-071.0", "069.0-119.0", "319.0-057.0", "331.0-057.0")
BASELINE = "133e35aec7b83b9d79888c46092f64d518e97299"


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


def compare(actual, reference):
    """Fixed elementwise combined gate; schema/dtype/discrete exact.

    Any nonfinite is a failure. No equal_nan shortcut is allowed here. If a
    pre-existing sentinel appears the failed path is saved for source audit.
    """
    ap, at = jax.tree_util.tree_flatten_with_path(actual)
    rp, rt = jax.tree_util.tree_flatten_with_path(reference)
    failures, max_abs, max_ratio = [], 0.0, 0.0
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
        elif not np.all(np.isfinite(a)) or not np.all(np.isfinite(r)):
            failures.append(
                {
                    "path": name,
                    "reason": "nonfinite",
                    "actual_count": int(np.count_nonzero(~np.isfinite(a))),
                    "reference_count": int(np.count_nonzero(~np.isfinite(r))),
                }
            )
        else:
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
        "nonfinite_policy": "fail",
    }


class Recorder:
    def __init__(self, output):
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.result = {"status": "running", "phases": [], "gates": [], "timing": {}}
        self.started = time.perf_counter()
        self.stop = threading.Event()
        self.peak = psutil.Process().memory_info().rss
        self.sampler = threading.Thread(target=self.sample, daemon=True)
        self.sampler.start()

    def sample(self):
        while not self.stop.wait(0.1):
            self.peak = max(self.peak, psutil.Process().memory_info().rss)

    def flush(self):
        self.result["peak_rss_bytes"] = self.peak
        self.result["elapsed_seconds"] = time.perf_counter() - self.started
        (self.output / "report.json").write_text(json.dumps(self.result, indent=2) + "\n", encoding="utf-8")

    def phase(self, name, **kwargs):
        entry = {
            "phase": name,
            "elapsed_seconds": time.perf_counter() - self.started,
            "rss_bytes": psutil.Process().memory_info().rss,
            **kwargs,
        }
        self.result["phases"].append(entry)
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self.flush()
        print(json.dumps(entry), flush=True)

    def gate(self, name, actual, reference):
        entry = {"name": name, **compare(actual, reference)}
        self.result["gates"].append(entry)
        self.flush()
        if not entry["passed"]:
            raise AssertionError(f"gate failed: {name}; see report.json")


def metadata(args):
    proc = psutil.Process()
    return {
        "command": sys.argv,
        "cwd": str(ROOT),
        "platform": platform.platform(),
        "python": sys.version,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "devices": [str(d) for d in jax.devices()],
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": proc.cpu_affinity(),
        "cpu_model": platform.processor(),
        "hostname": platform.node(),
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
            )
        },
        "cache_policy": "disabled"
        if os.environ.get("ORCHJAX_DISABLE_COMPILATION_CACHE") == "1"
        else "persistent_cache_may_hit",
        "gpu_count": sum(d.platform == "gpu" for d in jax.devices()),
        "allocated_cpu_cores": len(proc.cpu_affinity()),
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
    recorder.phase(name + "_compile_start")
    t = time.perf_counter()
    executable = lowered.compile()
    recorder.result["timing"][name + "_compile_seconds"] = time.perf_counter() - t
    recorder.phase(name + "_compiled")
    return executable


def stack_second(lanes, arguments):
    from dataclasses import replace

    return batch.stack_lanes([replace(lane, arguments=a) for lane, a in zip(lanes, arguments, strict=True)])


def benchmark(args, recorder, lanes, second):
    old_factory = legacy_factory(args.baseline_ref)
    first = lanes[0]
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
    scalar = lower_compile(batch.make_scalar_transition(first), first.arguments, "extracted_scalar", recorder)
    recorder.gate(
        "B1_original_AOT_vs_extracted_full_state_day_outputs", run_two(scalar, first.arguments, second[0]), old_results
    )
    # Independent per-point closures are a ground truth for the compatibility
    # audit itself, including any accidentally omitted context numeric value.
    scalars = [scalar]
    for lane in lanes[1:]:
        scalars.append(
            lower_compile(batch.make_scalar_transition(lane), lane.arguments, "scalar_" + lane.landpoint_id, recorder)
        )
    scalar_device = []
    references = []
    for fn, lane, s in zip(scalars, lanes, second, strict=True):
        t = time.perf_counter()
        device_pair = jax.device_put((lane.arguments, s))
        sync(device_pair)
        scalar_device.append(device_pair)
        recorder.result["timing"]["scalar_" + lane.landpoint_id + "_device_transfer_seconds"] = time.perf_counter() - t
        reference = run_two(fn, *device_pair)
        references.append(reference)
        with (args.output_dir / ("scalar_" + lane.landpoint_id + "_reference.pkl")).open("wb") as f:
            pickle.dump(jax.device_get(reference), f, protocol=pickle.HIGHEST_PROTOCOL)
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
            recorder.gate(f"B{size}_permutation", perm_result, jax.tree_util.tree_map(lambda v: v[order], actual))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/orchidee_man_250919.yaml")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--truth-root", type=Path, default=ROOT)
    parser.add_argument("--prepared-input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 4], choices=[1, 4])
    parser.add_argument("--warm-repeats", type=int, default=5)
    parser.add_argument("--baseline-ref", default=BASELINE)
    parser.add_argument("--mode", choices=["audit", "benchmark"], default="benchmark")
    args = parser.parse_args()
    if args.warm_repeats < 5:
        parser.error("at least five warm repeats required")
    args.output_dir = args.output_dir.resolve()
    os.environ.setdefault("ORCHIDEE_REPO_ROOT", str(ROOT))
    os.environ.setdefault("ORCHIDEE_DATA_ROOT", str(args.truth_root / "data"))
    os.environ.setdefault("ORCHIDEE_REFERENCE_ROOT", str(args.truth_root / "reference"))
    os.environ["ORCHIDEE_OUTPUT_ROOT"] = str(args.output_dir)
    os.environ.setdefault("ORCHJAX_DISABLE_COMPILATION_CACHE", "1")
    configure_jax_compilation_cache(ROOT)
    recorder = Recorder(args.output_dir)
    try:
        recorder.result["environment"] = metadata(args)
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
        recorder.result["status"] = "passed_audit_only" if args.mode == "audit" else "passed_bounded_forward_diagnostic"
    except Exception as e:
        recorder.result.update(
            status="failed", exception_type=type(e).__name__, message=str(e), traceback=traceback.format_exc()
        )
        recorder.phase("failed", exception_type=type(e).__name__, message=str(e))
        return 1
    finally:
        recorder.stop.set()
        recorder.flush()
    recorder.phase("complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
