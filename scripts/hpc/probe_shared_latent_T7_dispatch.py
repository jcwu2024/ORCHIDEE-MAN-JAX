"""Saved-only same-card block-each versus four-call dispatch comparison.

This entry can never lower/compile B1 or B4. Only one reduction helper is
compiled, reported separately; authorization and occupation ledger are external.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np

try:
    from scripts.hpc import probe_shared_latent_T7_batch as batch
except ImportError:
    import probe_shared_latent_T7_batch as batch

pair = batch.pair
try:
    from scripts.hpc import shared_latent_T7_dispatch_ad as dispatch
except ImportError:
    import shared_latent_T7_dispatch_ad as dispatch


def parser():
    p = pair.parser()
    p.description = __doc__
    p._option_string_actions["--stage"].choices = ("saved-dispatch",)
    p._option_string_actions["--horizon"].choices = (7,)
    p._option_string_actions["--pairs"].choices = (3,)
    p.set_defaults(wall_seconds=840)
    for name in ("training-prepared-manifest", "dataset-manifest", "cpu-readiness-report", "approval-report"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--load-seconds", type=float, default=360)
    p.add_argument("--b1-memory-report", type=Path)
    p.add_argument("--dry-run", action="store_true")
    return p


def launch_plan(args):
    if args.horizon != 7 or args.original_executable or not args.candidate_executable or args.validate_only:
        raise ValueError("saved candidate B1 T7 only; no original/model/validate-only fallback")
    if args.pairs != 3 or not 1 <= args.wall_seconds <= 840 or not 0 < args.load_seconds <= 360 or not 0 < args.call_seconds <= 180:
        raise ValueError("bounded dispatch deadlines and three pairs required")
    return {"schema_version": "latent_T7_dispatch_launch_plan_v1",
        "output": str(args.output), "candidate_executable": str(args.candidate_executable),
        "approval_report": str(args.approval_report), "requested_pairs": 3,
        "orders": [list(order) for order in dispatch.alternating_orders()],
        "model_compiles": {"B1": 0, "B4": 0}, "small_merge_compile_cap": 1,
        "wall_seconds": args.wall_seconds, "requires_independent_900_second_grant": True,
        "training_updates": 0, "GPU_executed": False}


def result_exit_code(report):
    """Exit3 records completed strict failure, never scientific acceptance."""
    if report.get("status") == "numerical_failed":
        return 3
    if report.get("status") == "passed" or ("status" not in report
            and report.get("schema_version") == "latent_T7_dispatch_launch_plan_v1"
            and report.get("GPU_executed") is False):
        return 0
    return 1


def final_receipt(report, output):
    """Print compact Python scalars; full NumPy metadata stays in saved JSON."""
    return {"status": report.get("status"), "report": str(output),
            "execution_completed": bool(report.get("execution_completed", False)),
            "process_exit_code": result_exit_code(report),
            "adoption": bool(report.get("adoption", False)),
            "scientific_acceptance": bool(report.get("scientific_acceptance", False))}


def run(args):
    plan = launch_plan(args)
    if args.dry_run:
        return plan
    dispatch.validate_scope(json.loads(args.approval_report.read_text()), output=args.output, wall_seconds=args.wall_seconds)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible.strip() != visible:
        raise ValueError("one externally allocated visible V100 required")
    if args.output.exists():
        raise FileExistsError(args.output)
    pair.adjacent("training_artifact_io")
    journal = pair.Journal(args.output, args.wall_seconds)
    journal.report.update(**plan, gpu_count=1, native_AD_argnums=[0, 1], helper_compiles=0,
        results=[], pairs=[], repeat_failures=[], numerical_failures=[], adoption=False,
        scientific_acceptance=False, optimizer_updates=0, rtol=1e-5, atol=1e-6)
    try:
        pair.adjacent("orcjax_gpu_runtime").initialize_gpu_backend()
        import jax
        from jax.experimental.serialize_executable import deserialize_and_load
        jax.config.update("jax_enable_x64", True)
        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu" or "V100" not in devices[0].device_kind:
            raise ValueError("exactly one V100 required")
        journal.report["runtime_preflight"] = pair.adjacent("probe_shared_union_micro4").validate_runtime_report(args.runtime_preflight, jax.__version__)
        journal.report["resources"] = {"pid": os.getpid(), "CUDA_VISIBLE_DEVICES": visible,
            "host_load": list(os.getloadavg()), "CPU_affinity": sorted(os.sched_getaffinity(0)),
            "device_kind": devices[0].device_kind}
        journal.emit("loading_exact_candidate_and_four_windows")
        source, owner, _, _, head, _, trees, identities = pair.assets(jax, args)
        parameters = trees["candidate"]
        selector = pair.adjacent("shared_latent_T7_batch_assets")
        with jax.default_device(jax.devices("cpu")[0]):
            selected = selector.load_training_T7_cases(jax, prepared_manifest=args.training_prepared_manifest,
                dataset_manifest=args.dataset_manifest, input_bundle=pair.read_manifest(args)["input_bundle"],
                head=head, method=identities["candidate"]["method"], count=4)
        identity = batch.mathematical_identity(jax, identities["candidate"], selected)
        readiness = json.loads(args.cpu_readiness_report.read_text())
        if readiness.get("status") != "CPU_readiness_passed" or readiness.get("mathematical_identity") != identity:
            raise ValueError("completed four-window CPU readiness differs")
        sidecar = json.loads(args.candidate_executable.with_suffix(".pkl.identity.json").read_text())
        if sidecar != {"identity": {**identities["candidate"], "device_kind": devices[0].device_kind}, "component": "T7"}:
            raise ValueError("exact saved B1 identity differs")
        journal.report.update(mathematical_identity=identity, window_metadata=selected["window_metadata"],
            B1_compiled_memory=batch.memory_planning(args, sidecar, readiness["shape_estimate"])["known_B1_compiled_memory"])
        abi = pair.adjacent("shared_full_state_ad_execution")
        singles = [(parameters, *case) for case in selected["cases"]]
        singles = [jax.device_put(single) for single in singles]
        began = time.monotonic()
        with journal.deadline(args.load_seconds, "load_saved_B1"), source.forbid_model_compile():
            saved = pair.read_pickle(args.candidate_executable)
            executable = deserialize_and_load(*saved)
            for single in singles:
                abi.validate_full_state_abi(jax, saved, executable, single, component="T7")
        journal.report["B1_load_seconds"] = time.monotonic() - began
        journal.emit("saved_B1_loaded_four_ABIs_validated", load_seconds=journal.report["B1_load_seconds"])
        del saved, trees
        gates = pair.adjacent("probe_shared_union_micro4")
        reference_paths = {}

        def preserve(mode, label, result, timing):
            host = jax.device_get(result)
            path = args.output.with_suffix("." + mode + "." + label + ".result.pkl")
            if path.exists():
                raise FileExistsError(path)
            owner.atomic_pickle(path, {"mathematical_identity": identity, "mode": mode, "label": label, "result": host})
            row = {**timing, "label": label, "artifact": str(path)}
            journal.report["results"].append(row)
            journal.emit("raw_arrays_and_timing_saved_before_gates", **row)
            row["validation"] = batch.batch_ad.validate_batch_result(jax, host, parameters, selected["stacked"][0], gates)
            row["JAX_memory_stats"] = devices[0].memory_stats()
            peak = row["JAX_memory_stats"].get("peak_bytes_in_use")
            if not isinstance(peak, (int, float)) or not np.isfinite(peak) or not 0 <= peak < 24 * 2**30:
                raise MemoryError("observed full serial/dispatch JAX peak exceeds24GiB or unavailable")
            if mode in reference_paths:
                comparison = batch.batch_ad.compare_full_results(jax, host, pair.read_pickle(reference_paths[mode])["result"], pair, head.contract)
                row["self_repeat"] = comparison
                if not comparison["passed"]:
                    journal.report["repeat_failures"].append({"mode": mode, "label": label, "comparison": comparison})
            else:
                reference_paths[mode] = path
            journal.emit("full_AD_gates_and_repeats_checked", mode=mode, label=label)
            return row

        # Full cold B1 outputs are preserved before building only the merge.
        cold = []
        for index, single in enumerate(singles):
            began = time.monotonic()
            with journal.deadline(args.call_seconds, "cold_B1_" + str(index)), source.forbid_model_compile():
                value = executable(*single)
                jax.block_until_ready(value)
            seconds = time.monotonic() - began
            journal.report["GPU_executed"] = True
            path = args.output.with_suffix(".cold-sample" + str(index) + ".result.pkl")
            if path.exists():
                raise FileExistsError(path)
            owner.atomic_pickle(path, {"window_metadata": selected["window_metadata"][index],
                "synchronized_full_AD_seconds": seconds, "result": jax.device_get(value)})
            journal.emit("cold_B1_full_arrays_and_timing_saved", sample=index, artifact=str(path), seconds=seconds)
            pair.validate_result(jax, jax.device_get(value), parameters, "candidate", selected["cases"][index][0])
            cold.append(value)
        began = time.monotonic()
        merge = jax.jit(lambda *rows: batch.batch_ad.merge_serial_results(jax, rows)).lower(*cold).compile()
        journal.report.update(helper_compiles=1, reduction_helper_compile_seconds=time.monotonic() - began)
        journal.emit("small_merge_helper_compiled", seconds=journal.report["reduction_helper_compile_seconds"])
        del cold

        def measure(mode, label):
            with journal.deadline(args.call_seconds, mode + "_" + label), source.forbid_model_compile():
                result, timing = dispatch.execute_four(executable, singles, merge, mode=mode, block=jax.block_until_ready)
            return preserve(mode, label, result, timing)

        for mode in dispatch.MODES:
            measure(mode, "warmup")
        for index, order in enumerate(dispatch.alternating_orders(args.pairs), 1):
            rows = [measure(mode, "pair" + str(index)) for mode in order]
            by_mode = {row["mode"]: row for row in rows}
            comparison = batch.batch_ad.compare_full_results(jax,
                pair.read_pickle(by_mode["dispatch_four"]["artifact"])["result"],
                pair.read_pickle(by_mode["block_each"]["artifact"])["result"], pair, head.contract)
            path = args.output.with_suffix(".pair" + str(index) + ".strict-comparison.json")
            owner.atomic_json(path, {"mathematical_identity": identity, "comparison": comparison})
            if not comparison["passed"]:
                journal.report["numerical_failures"].append({"pair": index, "comparison_artifact": str(path), "comparison": comparison})
            journal.report["pairs"].append({"pair": index, "order": list(order), "measurements": rows,
                "strict_passed": comparison["passed"], "comparison_artifact": str(path)})
            journal.emit("pair_completed", pair=index, strict_passed=comparison["passed"])
        medians = {mode: float(np.median([next(row["four_window_seconds"] for row in item["measurements"]
            if row["mode"] == mode) for item in journal.report["pairs"]])) for mode in dispatch.MODES}
        journal.report.update(status="numerical_failed" if journal.report["repeat_failures"] or journal.report["numerical_failures"] else "passed",
            execution_completed=True,
            GPU_executed=True, timing_summary={"four_window_medians": medians,
            "speedup": medians["block_each"] / medians["dispatch_four"], "complete_update_measured": False})
        journal.report["expected_process_exit_code"] = result_exit_code(journal.report)
        journal.emit("completed")
        return journal.report
    except Exception as error:
        journal.report.update(status="resource_failed" if isinstance(error, MemoryError) else "error",
            error_type=type(error).__name__, error=str(error))
        journal.emit("error")
        raise
    finally:
        journal.timer.cancel()


if __name__ == "__main__":
    args = parser().parse_args()
    report = run(args)
    print(json.dumps(report if args.dry_run else final_receipt(report, args.output), indent=2))
    raise SystemExit(result_exit_code(report))
