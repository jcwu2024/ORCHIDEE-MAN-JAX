"""Prepare the separately funded first-day B1/B4 transition JVP diagnostic."""
from __future__ import annotations
import argparse
import csv
import gzip
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

try:
    from scripts.hpc import probe_shared_latent_T7_state_jvp as prior
except ImportError:
    import probe_shared_latent_T7_state_jvp as prior
pair = prior.pair
PREFIX = "T7-first-day-jvp-20261009"
SCOPE = "latent_T7_first_day_jvp_20261009"
ROOT = Path("/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime")
CODE = ROOT / "checkouts/shared-split-width-v1"
OVERLAY = ROOT / "transfer" / PREFIX
DIAG = ROOT / "outputs/diagnostics/shared-latent-reader-20261004"


def parser():
    p = prior.parser()
    p.description = __doc__
    p.set_defaults(wall_seconds=4800, gpu_seconds_cap=5400, lower_seconds=180,
                   backend_seconds=1800, route_seconds=2100, route_mode="both")
    p.add_argument("--nvml-memory-log", type=Path,
        default=ROOT / "outputs/diagnostics" / ("latent-reader-T7-20261005-"+PREFIX+".memory.csv"))
    return p


def proposal_facts(output):
    return {"schema_version": "latent_T7_first_day_jvp_approval_v1", "funding_scope": SCOPE,
        "gpu_count": 1, "node": "gln01", "limit_gpu_seconds": 5400,
        "probe_wall_seconds": 4800, "protected_exit_reserve_seconds": 360,
        "lower_seconds": 180, "backend_seconds": 1800, "route_seconds": 2100,
        "runtime_preflight_seconds": 240, "model_compile_cap_per_route": 1,
        "new_model_graphs": 2, "jvp_calls": 24, "jvp_calls_per_route": 12,
        "alternating_cycles": 3, "optimizer_update_cap": 0, "saved_executable_load_count": 0,
        "worst_case_cost_cny": 0.0, "paid_fallback": False,
        "precision": "FP32_neural_FP64_physics", "sample_index": 3,
        "state_coordinates": [434, 2382, 2621, 200], "natural_transition_scale": True,
        "entry_script": (OVERLAY / "probe_shared_latent_T7_day_jvp.py").as_posix(),
        "working_directory": CODE.as_posix(), "output_report": Path(output).as_posix(),
        "approval_required_before_gpu_execution": True}


def component_module():
    try:
        from scripts.hpc import shared_latent_T7_day_jvp as component
    except ImportError:
        import shared_latent_T7_day_jvp as component
    return component


def first_day_readiness(jax, selected):
    component = component_module()
    b1 = component.first_day_inputs(jax, selected["cases"][3], batched=False)
    b4 = component.first_day_inputs(jax, selected["stacked"], batched=True)
    if np.shape(b1[0]) != (3854,) or np.asarray(b1[0]).dtype != np.dtype("float64"):
        raise ValueError("complete FP64 first-day state required")
    paths, tree = jax.tree_util.tree_flatten_with_path(b1)
    rows, batch_tree = jax.tree_util.tree_flatten(b4)
    if tree != batch_tree:
        raise ValueError("first-day B1/B4 static trees differ")
    receipt = []
    for (path, single), batched in zip(paths, rows, strict=True):
        single, batched = np.asarray(single), np.asarray(batched)
        if batched.shape != (4, *single.shape) or batched.dtype != single.dtype or not np.array_equal(single, batched[3], equal_nan=True):
            raise ValueError("first-day B1 sample3 versus B4 selected input differs")
        receipt.append({"path": jax.tree_util.keystr(path), "B1_shape": list(single.shape),
            "B4_shape": list(batched.shape), "dtype": str(single.dtype), "sample3_exact": True})
    return {"argument_count": 3, "argument_leaf_count": len(receipt), "sample3_exact": True,
        "full_state_width": 3854, "state_dtype": "float64", "leaves": receipt,
        "scope": "actual public first-day input selection only; no model trace"}


def sampled_nvml_memory(path):
    path = Path(path)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise MemoryError("sampled NVML memory receipt is empty")
    values = []
    for row in rows:
        epoch, used = float(row["epoch_seconds"]), float(row["memory_used_mib"])
        if not np.isfinite(epoch) or not np.isfinite(used) or used < 0:
            raise MemoryError("sampled NVML memory receipt has nonfinite/invalid values")
        values.append(used)
    peak = max(values)
    return {"path": str(path), "sample_count": len(values), "peak_mib": peak,
        "limit_mib": 24576, "passed": peak <= 24576,
        "scope": "sampled total visible-card memory, not an unsampled instantaneous peak"}


def validate_proposal(args, *, execution):
    record = json.loads(args.approval_report.read_text(encoding="utf-8"))
    approved_output = DIAG / (PREFIX + ".result.json")
    for key, expected in proposal_facts(approved_output).items():
        if type(record.get(key)) is not type(expected) or record.get(key) != expected:
            raise ValueError("first-day proposal scope differs: " + key)
    if execution:
        if record.get("approved") is not True:
            raise PermissionError("first-day GPU execution requires its own explicit human approval")
        if record.get("approval_origin") != "explicit_human_user" or not str(record.get("approval_note", "")).strip():
            raise PermissionError("first-day GPU execution requires human approval provenance")
        if args.output != approved_output:
            raise ValueError("approved output path differs")
    elif record.get("approved") is not False:
        raise ValueError("preparation must retain approved=false")
    return record


def validate_caps(args):
    if (args.route_mode != "both" or args.repair_record is not None or args.b4_jvp_executable is not None
        or args.horizon != 7 or args.gpu_seconds_cap != 5400 or args.wall_seconds != 4800
        or args.lower_seconds != 180 or args.backend_seconds != 1800 or args.route_seconds != 2100
        or args.force_compilation_parallelism != 4 or args.turnover_base_sha256 != pair.BASE_SHA
        or args.turnover_sha256 != pair.CORRECTED_SHA
        or args.runtime_preflight != DIAG / (PREFIX+".runtime-preflight.json")
        or args.nvml_memory_log != ROOT / "outputs/diagnostics" / ("latent-reader-T7-20261005-"+PREFIX+".memory.csv")):
        raise ValueError("fixed new first-day scope/caps required")


def _assets(jax, args):
    source, owner, transformer, context, head, _, trees, identities = pair.assets(jax, args)
    selector = pair.adjacent("shared_latent_T7_batch_assets")
    with jax.default_device(jax.devices("cpu")[0]):
        selected = selector.load_training_T7_cases(jax,
            prepared_manifest=args.training_prepared_manifest, dataset_manifest=args.dataset_manifest,
            input_bundle=pair.read_manifest(args)["input_bundle"], head=head,
            method=identities["candidate"]["method"], count=4)
    if list(selected["selected_artifact_ids"]) != [742, 275, 482, 299]:
        raise ValueError("exact four declared TRAIN windows required")
    identity = prior.batch_probe.mathematical_identity(jax, identities["candidate"], selected)
    saved = json.loads(args.b4_executable.with_suffix(args.b4_executable.suffix + ".identity.json").read_text())
    if saved.get("identity") != identity:
        raise ValueError("real four-window assets differ from completed B4 identity")
    meta = json.loads(args.state_input_metadata.read_text())
    digest = __import__("hashlib").sha256(np.ascontiguousarray(selected["cases"][3][0]).tobytes()).hexdigest()
    if digest != meta["identities"]["initial_state_sha256"]:
        raise ValueError("actual sample3 initial-state raw hash differs")
    return source, owner, transformer, context, head, trees["candidate"], selected, identity


def preserve_raw(owner, path, result, *, route, cycle, coordinate):
    prior._save(owner, path, {"route": route, "cycle": cycle, "coordinate": coordinate,
        "result": result, "tangent_scale": "natural_transition_no_div4"})


def run(args):
    validate_caps(args)
    validate_proposal(args, execution=args.stage == "execute" and not args.plan_only)
    if args.plan_only:
        return {"status": "plan_only", "GPU_executed": False, "model_traced": False,
            "planned_model_compiles": {"B1": 1, "B4": 1}, "planned_JVP_calls": 24,
            "JVP_calls": 0, "saved_executable_load_count": 0, "optimizer_updates": 0,
            "first_day_only": True, "precision": "FP32_neural_FP64_physics",
            "occupation_caps": proposal_facts(DIAG / (PREFIX + ".result.json"))}
    if args.output.exists():
        raise FileExistsError(args.output)
    pair.adjacent("training_artifact_io")
    if args.stage == "cpu-readiness":
        os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        ready = prior.cpu_metadata_readiness(args, None)
        import jax
        jax.config.update("jax_enable_x64", True)
        if any(device.platform != "cpu" for device in jax.local_devices()):
            raise ValueError("CPU readiness requires CPU backend")
        source, owner, _, _, _, _, selected, identity = _assets(jax, args)
        ready.update(schema_version="latent_T7_first_day_CPU_readiness_v1",
            scope=SCOPE, approved=False, model_traced=False, full_model_trace=False,
            actual_existing_window_arrays_checked=True, JAX_numeric_calls="bounded public input adapter only; no model",
            sampling=selected["window_metadata"], input_contract=selected["contract"],
            mathematical_identity=identity, saved_executable_load_count=0,
            first_day_argument_contract=first_day_readiness(jax, selected),
            runtime_preflight_action="new boundary requires one actual GPU preflight after approval")
        owner.atomic_json(args.output, ready)
        return ready
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("one explicitly approved GPU required")
    journal = pair.Journal(args.output, args.wall_seconds)
    journal.report.update(schema_version="latent_T7_first_day_jvp_v1", scope=SCOPE,
        GPU_executed=False, JVP_calls=0, model_compiles={"B1": 0, "B4": 0},
        optimizer_updates=0, training_updates=0, comparisons=[], raw_artifacts=[],
        saved_executable_load_count=0, precision="FP32_neural_FP64_physics",
        first_day_only=True, natural_transition_scale=True)
    try:
        pair.adjacent("orcjax_gpu_runtime").initialize_gpu_backend()
        import jax
        jax.config.update("jax_enable_x64", True)
        devices = jax.local_devices()
        if len(devices) != 1 or devices[0].platform != "gpu" or "V100" not in devices[0].device_kind:
            raise ValueError("one V100 required")
        journal.report.update(GPU_executed=True)
        journal.report["runtime_preflight"] = pair.adjacent("probe_shared_union_micro4").validate_runtime_report(args.runtime_preflight, jax.__version__)
        source, owner, transformer, context, head, parameters, selected, identity = _assets(jax, args)
        component = component_module()
        journal.report.update(sampling=selected["window_metadata"], mathematical_identity=identity)
        with ExitStack() as patches:
            from research.daily_coarse_graining.daily_operator import shared_process_head
            from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_recurrent_objective
            from jax_orchidee.stomate import carbon_kernels
            patches.enter_context(pair.install_turnover_overlay(carbon_kernels, args.turnover_overlay, pair.BASE_SHA))
            patches.enter_context(pair.candidate_backbone_patches(shared_process_head,
                transformer.shared_transformer_raw_outputs, owner.load_backbone_assembler(args.backbone_adapter)))
            rollout, _ = make_shared_recurrent_objective(head, context["template"], context["state_scales"], context["science_scales"])
            functions = dict(zip(("B1", "B4"), component.route_functions(jax, rollout.step), strict=True))
            directions = jax.device_put(prior.contract.direction_matrix(dtype=np.float64))
            prepared = {"B1": component.first_day_inputs(jax, selected["cases"][3], batched=False),
                        "B4": component.first_day_inputs(jax, selected["stacked"], batched=True)}
            parameters = jax.device_put(parameters)
            arguments = {route: (parameters, *jax.device_put(value)) for route, value in prepared.items()}
            def call_args(route, index):
                direction = directions[index, 3] if route == "B1" else directions[index]
                return (*arguments[route], direction)
            usage, executables, raw, timings = {"B1": 0.0, "B4": 0.0}, {}, {}, []
            @contextmanager
            def stage(route, label, cap=None):
                remaining = args.route_seconds-usage[route]
                if remaining <= 0:
                    raise TimeoutError(route + " route cap exhausted")
                began = time.monotonic()
                with journal.deadline(min(remaining, cap) if cap else remaining, label):
                    yield
                usage[route] += time.monotonic()-began
                journal.report["route_occupation_seconds"] = dict(usage)
            from jax.experimental.serialize_executable import serialize
            from scripts.hpc.run_shared_transformer_admission import memory_sizes
            for route in ("B1", "B4"):
                marker = args.output.with_suffix("."+route.lower()+".compile-attempt.json")
                if marker.exists():
                    raise FileExistsError(marker)
                attempt = {"route": route, "attempt": 1, "scope": SCOPE, "identity": identity,
                    "status": "lower_started", "lower_started_epoch": time.time(), "backend_started": False}
                owner.atomic_json(marker, attempt)
                lower_clock = time.monotonic()
                with stage(route, "lower_"+route, args.lower_seconds):
                    lowered = jax.jit(functions[route], compiler_options=identity["saved_B1_mathematical_identity"]["compiler_options"]).lower(*call_args(route, 0))
                lower_seconds = time.monotonic()-lower_clock
                attempt.update(status="backend_started", lower_seconds=lower_seconds,
                    lower_finished_epoch=time.time(), backend_started=True, backend_started_epoch=time.time())
                owner.atomic_json(marker, attempt)
                backend_clock = time.monotonic()
                with stage(route, "backend_"+route, args.backend_seconds):
                    executable = lowered.compile()
                backend_seconds = time.monotonic()-backend_clock
                attempt.update(status="compiled", backend_seconds=backend_seconds, backend_finished_epoch=time.time())
                owner.atomic_json(marker, attempt)
                journal.report["model_compiles"][route] = 1
                graph = args.output.with_suffix("."+route.lower()+".jvp.executable.pkl")
                with stage(route, "preserve_graph_"+route):
                    prior._save(owner, graph, serialize(executable))
                    owner.atomic_json(graph.with_suffix(graph.suffix+".identity.json"),
                        {"route": route, "identity": identity, "scope": SCOPE,
                         "public_step": "make_shared_recurrent_objective(...)[0].step", "natural_transition_scale": True})
                    journal.report.setdefault("saved_binary_artifacts", []).append(str(graph))
                    journal.emit("binary_saved_before_HLO_export", route=route, binary=str(graph))
                    hlo_path = args.output.with_suffix("."+route.lower()+".optimized-hlo.txt.gz")
                    if hlo_path.exists():
                        raise FileExistsError(hlo_path)
                    text = executable.as_text()
                    if not isinstance(text, str) or not text:
                        raise RuntimeError("optimized executable text unavailable after binary save")
                    with gzip.open(hlo_path, "wt", encoding="utf-8", compresslevel=3) as stream:
                        stream.write(text)
                    owner.atomic_json(hlo_path.with_suffix(hlo_path.suffix+".identity.json"),
                        {"route": route, "identity": identity, "scope": SCOPE,
                         "binary": str(graph), "evidence_kind": "actual optimized executable text; no causality claim"})
                    del text
                memory = memory_sizes(executable)
                journal.report.setdefault("compiled_graphs", []).append({"route": route, "path": str(graph),
                    "optimized_hlo": str(hlo_path), "memory": memory,
                    "lower_seconds": lower_seconds, "backend_seconds": backend_seconds})
                journal.emit("graph_preserved", route=route)
                if memory["estimated_live_bytes"] > 24*2**30:
                    raise MemoryError("compiled memory exceeds24GiB")
                nvml = sampled_nvml_memory(args.nvml_memory_log)
                journal.report.setdefault("sampled_NVML_memory", []).append(nvml)
                if not nvml["passed"]:
                    raise MemoryError("sampled NVML card memory exceeds24GiB")
                executables[route] = executable
            jax.block_until_ready((parameters, arguments, directions))
            for cycle in range(1, 4):
                for route in (("B1", "B4") if cycle % 2 else ("B4", "B1")):
                    for index, coordinate in enumerate(prior.contract.STATE_COORDINATES):
                        began = time.monotonic()
                        with stage(route, "JVP_"+route), source.forbid_model_compile():
                            result = executables[route](*call_args(route, index))
                            jax.block_until_ready(result)
                        seconds = time.monotonic()-began
                        with stage(route, "preserve_raw_"+route):
                            host = jax.device_get(result)
                            path = args.output.with_suffix(f".{route.lower()}.cycle{cycle}.coordinate{coordinate}.result.pkl")
                            preserve_raw(owner, path, host, route=route, cycle=cycle, coordinate=coordinate)
                        raw[(route, cycle, coordinate)] = host
                        journal.report["raw_artifacts"].append(str(path))
                        journal.report["JVP_calls"] += 1
                        timings.append({"route": route, "cycle": cycle, "coordinate": coordinate, "seconds": seconds})
                        stats = devices[0].memory_stats()
                        peak = stats.get("peak_bytes_in_use") if isinstance(stats, dict) else None
                        journal.report.setdefault("observed_memory", []).append({"route": route, "cycle": cycle, "coordinate": coordinate, "stats": stats})
                        journal.emit("raw_JVP_preserved", route=route, cycle=cycle, coordinate=coordinate)
                        if not isinstance(peak, (int, float)) or not 0 <= peak <= 24*2**30:
                            raise MemoryError("observed memory unavailable or exceeds24GiB")
            # All24 complete arrays exist before the first numerical comparison.
            nvml = sampled_nvml_memory(args.nvml_memory_log)
            journal.report.setdefault("sampled_NVML_memory", []).append(nvml)
            if not nvml["passed"]:
                raise MemoryError("sampled NVML card memory exceeds24GiB")
            for cycle in range(1, 4):
                for coordinate in prior.contract.STATE_COORDINATES:
                    comparison = component.compare_results(jax, raw[("B1", cycle, coordinate)], raw[("B4", cycle, coordinate)], sample_index=3)
                    journal.report["comparisons"].append({"cycle": cycle, "coordinate": coordinate, "routes": comparison,
                        "same_route_repeats": {route: component.compare_results(jax, raw[(route, cycle, coordinate)], raw[(route, 1, coordinate)]) for route in ("B1", "B4")}})
            passed = all(row["routes"]["passed"] and all(value["passed"] for value in row["same_route_repeats"].values()) for row in journal.report["comparisons"])
            journal.report.update(status="passed" if passed else "numerical_failed", JVP_call_timings=timings,
                median_call_seconds={route: float(np.median([row["seconds"] for row in timings if row["route"] == route])) for route in ("B1", "B4")},
                timing_scope="diagnostic_only_not_speed_acceptance", inference="first_day_localization_only_not_T7_causality_or_adoption")
            journal.emit("complete_arrays_and_comparisons_preserved")
            return journal.report
    except Exception as error:
        journal.report.update(status="error", error_type=type(error).__name__, error=str(error))
        journal.emit("error")
        raise
    finally:
        journal.timer.cancel()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__, "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({key: result.get(key) for key in ("status", "GPU_executed", "model_compiles", "JVP_calls", "planned_JVP_calls", "saved_executable_load_count", "optimizer_updates")}))
    return 3 if result.get("status") == "numerical_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
