"""No-compilation original full-state AD control for exact step1 micro4 inputs.

Load the existing admitted immutable original binary. Preserve full old/new
results for forward/parameter/state regression against corrected artifacts.
No optimizer, new model graph, source patch or neural replay is introduced.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

try:
    from scripts.hpc.probe_shared_leaf_full_state_cuda import adjacent, prepare, Journal, BASE_SHA, CORRECTED_SHA, read_pickle, validate_snapshot, compare_repeats, sha
except ModuleNotFoundError:
    from probe_shared_leaf_full_state_cuda import adjacent, prepare, Journal, BASE_SHA, CORRECTED_SHA, read_pickle, validate_snapshot, compare_repeats, sha


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "prepared-context", "old-snapshot", "new-snapshot", "field-adapter", "backbone-adapter", "turnover-overlay", "runtime-preflight", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--turnover-base-sha256", default=BASE_SHA)
    p.add_argument("--turnover-sha256", "--turnover-corrected-sha256", dest="turnover_sha256", default=CORRECTED_SHA)
    p.add_argument("--original-executable", type=Path, help="default: manifest components.daily.executable; must match admitted full-state binary")
    p.add_argument("--corrected-output", type=Path, help="corrected JSON output stem; compare both existing repeat1 complete result artifacts")
    p.add_argument("--wall-seconds", type=float, default=180)
    p.add_argument("--call-seconds", type=float, default=60)
    return p


def run(args):
    if not 1 <= args.wall_seconds <= 1020 or not 0 < args.call_seconds <= args.wall_seconds:
        raise ValueError("bounded remaining reserve wall≤1020 and call≤wall required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible.strip() != visible:
        raise ValueError("parent must explicitly allocate exactly one visible GPU")
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adjacent("training_artifact_io")
    journal = Journal(args.output, args.wall_seconds)
    journal.report.update(schema_version="shared_leaf_original_saved_full_state_control_v1", model_compiles=0,
        scope="immutable_original_binary_old_new_step1_artifact275_rows16_20_control")
    try:
        adjacent("orcjax_gpu_runtime").initialize_gpu_backend()
        import jax
        jax.config.update("jax_enable_x64", True)
        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu" or "V100" not in devices[0].device_kind:
            raise ValueError("exactly one allocated V100 required")
        adjacent("probe_shared_union_micro4").validate_runtime_report(args.runtime_preflight, jax.__version__)
        journal.emit("original_control_input_validation")
        source, owner, manifest, reference, _, samples, identity = prepare(jax, args)
        identity["original_control_helper_sha256"] = sha(Path(__file__))
        path = args.original_executable or Path(manifest["components"]["daily"]["executable"])
        admission = json.loads(Path(manifest["components"]["daily"]["report"]).read_text())
        journal.report.update(identity=identity, original_executable=str(path))
        abi = adjacent("shared_full_state_ad_execution")
        device_samples = jax.device_put(samples)
        executable = None
        for schedule in ("old", "new"):
            saved = read_pickle(getattr(args, schedule+"_snapshot"))
            parameters, _ = validate_snapshot(jax, saved, manifest, schedule)
            del saved
            device_parameters = jax.device_put(parameters)
            arguments = (device_parameters, device_samples.inputs.continuous_state, device_samples)
            if executable is None:
                journal.emit("original_admitted_full_state_binary_loading")
                executable, evidence = abi.load_full_state_executable(jax=jax, path=path, arguments=arguments,
                    component="daily", admission=admission, architecture="original", expected_reference=reference,
                    compile_guard=source.forbid_model_compile)
                journal.report["original_full_state_admission"] = evidence
            journal.emit("original_control_call_entered", schedule=schedule)
            began = time.monotonic()
            with journal.deadline(args.call_seconds, "original_control_"+schedule), source.forbid_model_compile():
                host = jax.device_get(jax.block_until_ready(executable(*arguments)))
            output = args.output.with_suffix("."+schedule+".original.result.pkl")
            if output.exists(): raise FileExistsError(output)
            owner.atomic_pickle(output, {"identity": identity, "schedule": schedule, "result": host})
            # Complete result is preserved before any comparison or validation.
            if not isinstance(host, (tuple, list)) or len(host) != 2 or not isinstance(host[1], (tuple, list)) or len(host[1]) != 2:
                raise ValueError("original result is not full parameter and state AD")
            if jax.tree_util.tree_structure(host[1][0]) != jax.tree_util.tree_structure(parameters) or host[1][1].shape != (4, 3854):
                raise ValueError("original parameter/state gradient ABI differs")
            row = {"schedule": schedule, "artifact": str(output), "seconds": time.monotonic()-began,
                   "JAX_device_memory_stats": devices[0].memory_stats()}
            if args.corrected_output:
                corrected_path = args.corrected_output.with_suffix("."+schedule+".repeat1.result.pkl")
                corrected = read_pickle(corrected_path)
                if corrected.get("schedule") != schedule:
                    raise ValueError("corrected result schedule differs")
                corrected_identity = corrected["identity"]
                for key in ("source_git_head", "manifest_sha256", "input_binding", "snapshots", "parameters", "parameter_leaves", "precision", "loss", "turnover_sha256", "sources"):
                    if corrected_identity.get(key) != identity.get(key):
                        raise ValueError("corrected input/source identity differs: "+key)
                current = corrected["result"]
                # Original binary auxiliary averages state diagnostics or omits
                # them. Compare forward/gates on the shared ordinary aux only.
                def ordinary(value):
                    return (value[0][0], {k: v for k, v in value[0][1].items() if k != "state_diagnostics"})
                row["forward_and_gates_regression"] = compare_repeats(jax, ordinary(host), ordinary(current))
                row["parameter_gradient_regression"] = compare_repeats(jax, host[1][0], current[1][0])
                row["state_gradient_change"] = compare_repeats(jax, host[1][1], current[1][1])
                row["corrected_artifact"] = str(corrected_path)
                del corrected, current
            journal.report["results"].append(row)
            journal.emit("original_control_result_preserved", **row)
            del host, arguments, device_parameters, parameters
        journal.report["status"] = "completed"
        if args.corrected_output:
            journal.report["strict_forward_parameter_regression_passed"] = all(
                row[key]["strict_parity"] for row in journal.report["results"]
                for key in ("forward_and_gates_regression", "parameter_gradient_regression"))
        journal.emit("completed")
        return journal.report
    except Exception as error:
        journal.report.update(status="error", error_type=type(error).__name__, error=str(error))
        journal.emit("error")
        raise
    finally:
        journal.timer.cancel()


if __name__ == "__main__":
    run(parser().parse_args())
