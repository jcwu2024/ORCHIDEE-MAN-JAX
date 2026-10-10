"""Trace corrected saved T30 binaries; never lower or compile a model."""
from __future__ import annotations

import io
import json
import re
from pathlib import Path
import time

import numpy as np

import probe_shared_latent_reader_T7_pair as pair


def main():
    parser = pair.parser()
    parser.add_argument("--profile-architecture", choices=("original", "candidate", "both"), default="both")
    parser.add_argument("--index-seconds", type=float, default=150)
    args = parser.parse_args()
    if args.stage != "matched" or args.horizon != 30 or args.validate_only:
        raise ValueError("saved corrected T30 matched profile only")
    if not args.original_executable or not args.candidate_executable:
        raise ValueError("both saved binaries required; no compile fallback")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    journal = pair.Journal(args.output, args.wall_seconds)
    journal.report.update(scope="corrected_saved_T30_profile_20261008",
        model_compiles={"original": 0, "candidate": 0}, optimizer_updates=0,
        variants={}, scientific_acceptance=False, inherited_state_failures_retained=True)
    try:
        jax, devices = pair.adjacent("orcjax_gpu_runtime").initialize_gpu_backend()
        jax.config.update("jax_enable_x64", True)
        if len(devices) != 1 or "V100" not in devices[0].device_kind:
            raise ValueError("one V100 required")
        pair.adjacent("probe_shared_original_full_state_schedule").load_adjacent("training_artifact_io")
        journal.emit("loading_assets")
        source, owner, _, context, _, case, parameters, identities = pair.assets(jax, args)
        abi = pair.adjacent("shared_full_state_ad_execution")
        from jax.experimental.serialize_executable import deserialize_and_load
        device_case = jax.device_put(case)
        architectures = ("original", "candidate") if args.profile_architecture == "both" else (args.profile_architecture,)
        for architecture in architectures:
            identities[architecture]["device_kind"] = devices[0].device_kind
            path = getattr(args, architecture+"_executable")
            sidecar = json.loads(path.with_suffix(path.suffix+".identity.json").read_text())
            if sidecar != {"identity": identities[architecture], "component": "T30"}:
                raise ValueError("saved identity differs: "+architecture)
            arguments = (jax.device_put(parameters[architecture]), *device_case)
            began = time.monotonic()
            journal.emit("loading_binary", architecture=architecture)
            with journal.deadline(360, "load_"+architecture), source.forbid_model_compile():
                serialized = pair.read_pickle(path)
                executable = deserialize_and_load(*serialized)
                abi.validate_full_state_abi(jax, serialized, executable, arguments, component="T30")
            del serialized
            row = {"executable": str(path), "load_seconds": time.monotonic()-began,
                   "identity": identities[architecture]}
            journal.report["variants"][architecture] = row
            journal.emit("binary_loaded", architecture=architecture, load_seconds=row["load_seconds"])
            def call():
                with journal.deadline(args.call_seconds, "call_"+architecture), source.forbid_model_compile():
                    result = executable(*arguments)
                    jax.block_until_ready(result)
                return result

            journal.emit("cold_call", architecture=architecture)
            result = call()
            baseline = jax.device_get(result)
            row["validation"] = pair.validate_result(jax, baseline, parameters[architecture], architecture, case[0])
            del result
            row["warm_seconds"] = []
            for _ in range(3):
                began = time.monotonic()
                result = call()
                row["warm_seconds"].append(time.monotonic()-began)
                del result
            row["warm_median_seconds"] = float(np.median(row["warm_seconds"]))
            trace_dir = args.output.with_suffix(".traces") / architecture
            journal.emit("profiling", architecture=architecture, warm_seconds=row["warm_seconds"])
            began = time.monotonic()
            with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
                with jax.profiler.TraceAnnotation("corrected_saved_T30_"+architecture):
                    result = call()
            row["profile_with_start_stop_seconds"] = time.monotonic()-began
            row["trace_directory"] = str(trace_dir)
            host = jax.device_get(result)
            comparison = pair.repeat_comparison(jax, host, baseline, context["head"].contract)
            row["profile_repeat"] = {
                "forward_and_gates": comparison["forward_and_gates"],
                "all_parameter_gradients": comparison["all_parameter_gradients"],
                "all_initial_state_gradients": comparison["all_initial_state_gradients"],
                "state_failure_coordinates": comparison["state_failure_coordinates"]}
            row["trace_files"] = [str(p) for p in trace_dir.rglob("*.trace.json.gz")]
            row["memory_stats"] = devices[0].memory_stats()
            if row["memory_stats"].get("peak_bytes_in_use", 0) >= 24*2**30:
                raise MemoryError("24GiB exceeded")
            del host, baseline, result
            journal.emit("profile_saved", architecture=architecture)
            from analyze_shared_execution_trace import hlo_gemm_shapes
            with journal.deadline(args.index_seconds, "saved_hlo_index_"+architecture), source.forbid_model_compile():
                text = executable.as_text()
                records, ambiguous, shapes = {}, set(), {}
                for line in io.StringIO(text):
                    instruction = re.match(r"\s*(?:ROOT )?%([^ ]+) =", line)
                    if not instruction:
                        continue
                    name = instruction[1]
                    location = re.search(r'source_file="([^"]+)" source_line=(\d+)', line)
                    if location:
                        value = {"file": location[1], "line": int(location[2])}
                        if name in records and records[name] != value:
                            ambiguous.add(name)
                        records[name] = value
                    if 'custom_call_target="__cublas' in line:
                        shapes.update(hlo_gemm_shapes(line))
                for name in ambiguous:
                    records.pop(name, None)
                index = args.output.with_suffix("."+architecture+".hlo-index.json")
                owner.atomic_json(index, {"sources": records, "gemm_shapes": shapes,
                    "ambiguous_instruction_names": sorted(ambiguous), "saved_executable_only": True})
                row.update(hlo_index_file=str(index), hlo_text_characters=len(text),
                           source_record_count=len(records), gemm_shape_count=len(shapes))
                del text
            del executable, arguments
            journal.emit("index_saved", architecture=architecture)
        journal.report["status"] = "profile_completed_inherited_numerical_failures"
        journal.emit("completed")
    except Exception as error:
        journal.report.update(status="error", error_type=type(error).__name__, error=str(error))
        journal.emit("error", status="error", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
