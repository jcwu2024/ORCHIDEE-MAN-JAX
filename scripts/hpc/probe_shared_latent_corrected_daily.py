"""One corrected candidate daily graph and saved-only same-card full AD pairs.

Uses the unchanged public mean4 loss and native parameter/initial-state AD.
The accepted turnover overlay retains Fortran provenance. No optimizer updates,
cross-architecture numerical equality, training or scientific adoption occur.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

try:
    from scripts.hpc.probe_shared_leaf_full_state_cuda import adjacent, read_pickle, sha, Journal, BASE_SHA, CORRECTED_SHA
except ModuleNotFoundError:
    from probe_shared_leaf_full_state_cuda import adjacent, read_pickle, sha, Journal, BASE_SHA, CORRECTED_SHA

SOURCE_HEAD = "c599236bcf57b745e9bafed678298010f4c3b58f"
METHOD_SHA = "2543969418d524f07985053997ea9830c2afda02688b181ed033feab98b8c5ed"
CANDIDATE = "shared_latent32x384_reader64x2"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "source-profile", "prepared-context", "parameters", "field-adapter", "backbone-adapter", "transformer-module", "latent-module", "turnover-overlay", "runtime-preflight", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--validation-output", type=Path, help="CPU-only evidence path; never writes the future GPU report")
    p.add_argument("--saved-executable", type=Path, help="explicit saved-only recovery; never compiles")
    p.add_argument("--wall-seconds", type=float, default=6840)
    p.add_argument("--lower-seconds", type=float, default=600)
    p.add_argument("--backend-seconds", type=float, default=4800)
    p.add_argument("--call-seconds", type=float, default=180)
    p.add_argument("--pairs", type=int, choices=(3,), default=3)
    p.add_argument("--force-compilation-parallelism", type=int, choices=(4,), default=4)
    return p


def preparation_paths(args):
    """Reject absent immutable assets before backend setup, without reading arrays."""
    manifest = adjacent("probe_shared_latent_dual").read_manifest(args.manifest)
    if manifest.get("admission_mode") != "corrected_performance_diagnostic":
        raise ValueError("explicit corrected performance manifest required")
    item = manifest["architectures"]["candidate"]["daily"]
    if str(args.output) != item["report"] or str(args.output.with_suffix(".executable.pkl")) != item["executable"]:
        raise ValueError("candidate daily unique manifest filenames differ")
    if str(args.parameters) != manifest["candidate_parameters"]:
        raise ValueError("candidate initialization filename differs")
    original = manifest["architectures"]["original"]["daily"]
    required = [args.source_profile, args.prepared_context, args.parameters, args.field_adapter,
                args.backbone_adapter, args.transformer_module, args.latent_module, args.turnover_overlay,
                manifest["input_bundle"], manifest["original_checkpoint"], original["report"],
                original["executable"], original["executable"] + ".identity.json", original["verification_report"]]
    if args.saved_executable:
        required += [args.saved_executable, str(args.saved_executable) + ".identity.json"]
        if str(args.saved_executable) != item["executable"]:
            raise ValueError("saved-only recovery must use the manifest's sole candidate graph")
    if args.validate_only and (args.validation_output is None or args.validation_output == args.output or args.validation_output.exists()):
        raise ValueError("CPU validation requires a fresh separate --validation-output")
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(json.dumps({"missing_assets": missing}))
    if not 1 <= args.wall_seconds <= 6840 or not 0 < args.lower_seconds <= 600 or not 0 < args.backend_seconds <= 4800 or not 0 < args.call_seconds <= 180:
        raise ValueError("bounded daily deadlines differ")
    return manifest


def assets(jax, args, manifest):
    """Reuse public prepared context and independently saved original/candidate trees."""
    loader = adjacent("probe_shared_latent_reader")
    package = "research.daily_coarse_graining.daily_operator."
    transformer = loader.load(args.transformer_module, package + "shared_transformer")
    latent = loader.load(args.latent_module, package + "shared_latent_reader")
    source = adjacent("probe_shared_original_full_state_schedule")
    source.load_overlays()
    owner = adjacent("probe_shared_expert_real_daily")
    pair = adjacent("probe_shared_latent_reader_T7_pair")
    profile, reference = source.read_reference({"source_profile": str(args.source_profile)})
    root = Path(os.environ.get("ORCHIDEE_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    method = reference["method"]
    method_digest = hashlib.sha256(json.dumps(method, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if head != SOURCE_HEAD or reference["source_git_head"] != head or method_digest != METHOD_SHA or manifest["reference"] != {"source_git_head": head, "method": method}:
        raise ValueError("immutable public source/method differs")
    diagnostic = manifest["performance_diagnostic"]
    if diagnostic["turnover_sha256"] != CORRECTED_SHA or diagnostic["method_sha256"] != METHOD_SHA or sha(args.turnover_overlay) != CORRECTED_SHA:
        raise ValueError("accepted guarded turnover/method differs")
    context = read_pickle(args.prepared_context)
    expected = owner.context_identity(reference, Path(manifest["original_checkpoint"]), 742)
    if context["identity"] != expected or Path(context["daily_executable"]).name != Path(profile["executables"]["daily"]["executable"]).name:
        raise ValueError("prepared artifact742/update0/micro4 context differs")
    bundle = read_pickle(manifest["input_bundle"])
    if bundle.get("profile_sha256") != sha(args.source_profile):
        raise ValueError("compact input bundle is not bound to the declared source profile")
    workload = source.validate_bundle(jax, bundle, manifest, reference)
    if int(bundle["artifact_ids"][0]) != 742:
        raise ValueError("first declared training artifact must be742")
    samples = jax.tree_util.tree_map(lambda value: np.asarray(value)[:4].copy(), bundle["payloads"][0]["daily"])
    if pair.tree_digest(jax, samples) != pair.tree_digest(jax, context["daily4"]):
        raise ValueError("prepared micro4 differs from actual first bundle micro4")
    config = context["head"].shared_config
    if (config.width, config.backbone_blocks, config.coordinate_blocks, config.output_chunk_size) != (768, 8, 2, 64) or config.decoder_execution != "gather_dynamic_reuse":
        raise ValueError("full original architecture differs")
    candidate_head = replace(context["head"], shared_config=latent.shared_latent_reader_config(config))
    if candidate_head.shared_config.latent_reader_execution != "field" or candidate_head.shared_config.output_chunk_size != 64 or (latent.LATENT_SLOTS, latent.LATENT_LAYERS, latent.READER_WIDTH, latent.READER_LAYERS) != (32, 2, 64, 2):
        raise ValueError("fixed latent32/private64x2 field/chunk64 configuration differs")
    identity = {"source_head": head, "artifact_id": 742, "input_source_method": method,
                "seed": int(method["daily_protocol"]["seed"]), "candidate_config": vars(candidate_head.shared_config),
                "candidate_source_sha256": {"shared_transformer.py": sha(args.transformer_module), "shared_latent_reader.py": sha(args.latent_module)},
                "source_reuse_assembler_sha256": sha(args.backbone_adapter),
                "source_reuse_projection_sha256": sha(args.backbone_adapter.with_name("shared_backbone_projection_reuse.py"))}
    saved = read_pickle(args.parameters)
    if saved["identity"] != identity:
        raise ValueError("saved fresh candidate full-tree identity differs")
    from scripts.hpc.run_shared_learning import load_learning_checkpoint
    checkpoint = load_learning_checkpoint(Path(manifest["original_checkpoint"]), expected_method=method)
    if checkpoint["completed_updates"] != 0:
        raise ValueError("original initialization must be update0")
    trees = {"original": checkpoint["parameters"], "candidate": saved["parameters"]}
    for name, tree in trees.items():
        pair.parameter_gate(jax, tree, name)
    abi = adjacent("shared_full_state_ad_execution")
    original_item = manifest["architectures"]["original"]["daily"]
    original_sidecar = json.loads(Path(original_item["executable"] + ".identity.json").read_text())
    original_view = abi.validate_corrected_leaf_daily_diagnostic(
        json.loads(Path(original_item["report"]).read_text()), json.loads(Path(original_item["verification_report"]).read_text()), original_sidecar,
        expected_reference=manifest["reference"], expected_turnover_sha256=CORRECTED_SHA,
        expected_method_sha256=METHOD_SHA, expected_executable=original_item["executable"],
        expected_prepared_context_identity=diagnostic["prepared_context_identity"])
    candidate_graph_identity = {"source_git_head": head, "method": method, "turnover_base_sha256": BASE_SHA,
        "turnover_sha256": CORRECTED_SHA, "native_AD_argnums": [0, 1], "training_updates": 0,
        "precision": "FP32_neural_FP64_physics", "architecture": CANDIDATE, "parameter_count": 66123713,
        "parameter_leaves": 355, "prepared_context_identity": diagnostic["prepared_context_identity"],
        "prepared_context_source_identity": expected, "input_bundle": manifest["input_bundle"],
        "input_case_sha256": pair.tree_digest(jax, samples),
        "parameter_provenance": {"artifact": str(args.parameters), "identity": identity, "initialization": "fresh_public_whole_tree"},
        "loss": "unchanged_public_joint_daily_mean4", "state_diagnostics": "unreduced_four_rows_observation_only",
        "compiler_options": {"xla_gpu_enable_llvm_module_compilation_parallelism": True, "xla_gpu_force_compilation_parallelism": 4}}
    return source, owner, pair, transformer, context, candidate_head, samples, trees, abi.json_identity(candidate_graph_identity), original_view, workload


def compare_repeats(jax, actual, reference, contract):
    helper = adjacent("inspect_shared_leaf_cuda_acceptance")
    def ordinary(value):
        return value[0], {key: item for key, item in value[1].items() if key != "state_diagnostics"}
    forward = helper.compare_tree(jax, ordinary(actual[0]), ordinary(reference[0]))
    diagnostics = helper.inspect_diagnostics(jax, actual[0][1]["state_diagnostics"], reference[0][1]["state_diagnostics"])
    def actual_rows(value):
        if isinstance(value, dict):
            if "artifact_row" in value:
                value["artifact_row"] -= 16
            for child in value.values():
                actual_rows(child)
        elif isinstance(value, list):
            for child in value:
                actual_rows(child)
    actual_rows(diagnostics)
    state = helper.inspect_state(actual[1][1], reference[1][1], contract)
    # This protocol uses artifact742 rows0:4, not the leaf probe's rows16:20.
    for failure in state["failures"]:
        failure["artifact_row"] = failure["micro4_row"]
    return {"forward_and_gates": {"passed": forward["passed"] and diagnostics["passed"], "ordinary": forward, "row_diagnostics": diagnostics},
            "all_parameter_gradients": helper.compare_tree(jax, actual[1][0], reference[1][0]),
            "all_initial_state_gradients": state}


def run(args):
    manifest = preparation_paths(args)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.validate_only:
        os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        import jax
    else:
        jax, devices = adjacent("orcjax_gpu_runtime").initialize_gpu_backend()
        if len(devices) != 1 or devices[0].platform != "gpu" or "V100" not in devices[0].device_kind:
            raise ValueError("exactly one free-node V100 required")
    jax.config.update("jax_enable_x64", True)
    values = assets(jax, args, manifest)
    source, owner, pair, transformer, context, candidate_head, samples, trees, identity, original_view, workload = values
    if args.validate_only:
        result = {"status": "assets_validated", "GPU_executed": False, "model_compiles": 0, "optimizer_updates": 0,
                  "scientific_acceptance": False, "candidate_identity": identity,
                  "original_daily_source_view": original_view, "workload": workload}
        owner.atomic_json(args.validation_output, result)
        return result
    preflight = adjacent("probe_shared_union_micro4").validate_runtime_report(args.runtime_preflight, jax.__version__)
    journal = Journal(args.output, args.wall_seconds)
    journal.report.update(schema_version="shared_corrected_latent_reader_daily_pair_v1", component="daily", training_updates=0,
        precision="FP32_neural_FP64_physics", native_AD_argnums=[0, 1], gpu_count=1, runtime_preflight=preflight,
        model_compiles={"original": 0, "candidate": 0}, model_compiles_scope="saved-only matched segment",
        occupation_model_compiles={"original": 0, "candidate": 0}, cold_measurements={}, pairs=[], numerical_failures=[],
        complete_AD_calls_attempted=0, complete_AD_calls_completed=0, optimizer_updates=0,
        identities={"candidate": identity, "original": original_view["identity"]}, original_source_view=original_view,
        inherited_source_strict_state_failures=[{"architecture": "original", "component": "daily", **failure}
            for failure in original_view["state_repeat_failure_records"]],
        source_corrected_vs_original_regression=original_view["source_corrected_vs_original_regression"],
        cross_architecture_parity_is_gate=False, runner_harness_sha256=sha(Path(__file__)))
    try:
        import jax.numpy as jnp
        from jax.experimental.serialize_executable import serialize, deserialize_and_load
        from scripts.hpc.run_shared_transformer_admission import memory_sizes
        abi = adjacent("shared_full_state_ad_execution")
        leaf = adjacent("probe_shared_leaf_full_state_cuda")
        device_samples = jax.device_put(samples)
        arguments = {name: (jax.device_put(tree), device_samples.inputs.continuous_state, device_samples) for name, tree in trees.items()}
        candidate_path = Path(manifest["architectures"]["candidate"]["daily"]["executable"])
        if not args.saved_executable:
            if candidate_path.exists() or Path(str(candidate_path) + ".identity.json").exists():
                raise FileExistsError("sole candidate graph already exists; explicit saved-only recovery required")
            from research.daily_coarse_graining.daily_operator import shared_process_head
            from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_joint_daily_objective
            from jax_orchidee.stomate import carbon_kernels
            micro, dual = adjacent("shared_daily_microbatch"), adjacent("shared_daily_dual_ad")
            with ExitStack() as patches:
                patches.enter_context(leaf.install_turnover_overlay(carbon_kernels, args.turnover_overlay, BASE_SHA))
                patches.enter_context(pair.candidate_backbone_patches(shared_process_head, transformer.shared_transformer_raw_outputs, owner.load_backbone_assembler(args.backbone_adapter)))
                patches.enter_context(leaf.retain_row_diagnostics(micro))
                scalar = make_shared_joint_daily_objective(context["runtime"], candidate_head, context["template"], context["state_scales"], context["science_scales"], include_state_diagnostics=True)
                objective = dual.make_daily_dual_objective(jax, micro.make_microbatch_objective(jax, jnp, scalar))
                backward = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True), compiler_options=identity["compiler_options"])
                journal.emit("candidate_lowering")
                began = time.monotonic()
                with journal.deadline(args.lower_seconds, "lower"):
                    lowered = backward.lower(*arguments["candidate"])
                lower_seconds = time.monotonic() - began
                journal.emit("candidate_backend", lower_seconds=lower_seconds)
                began = time.monotonic()
                journal.report["occupation_model_compiles"]["candidate"] = 1
                with journal.deadline(args.backend_seconds, "backend"):
                    compiled = lowered.compile()
                backend_seconds = time.monotonic() - began
                # First postcompile work preserves the graph before any validation.
                owner.atomic_pickle(candidate_path, serialize(compiled))
                owner.atomic_json(Path(str(candidate_path) + ".identity.json"), {"component": "daily", "identity": identity})
                owner.atomic_json(args.output.with_suffix(".compile.json"), {"status": "graph_saved", "scientific_acceptance": False,
                    "model_compiles": {"original": 0, "candidate": 1}, "lower_seconds": lower_seconds,
                    "backend_seconds": backend_seconds, "executable": str(candidate_path), "identity": identity})
                del compiled, lowered, backward
        executables = {}
        for name in ("original", "candidate"):
            path = Path(manifest["architectures"][name]["daily"]["executable"])
            began = time.monotonic()
            sidecar = json.loads(Path(str(path) + ".identity.json").read_text())
            expected_identity = identity if name == "candidate" else original_view["identity"]
            if sidecar != {"component": "daily", "identity": expected_identity}:
                raise ValueError("saved daily graph sidecar exact identity differs")
            with source.forbid_model_compile():
                serialized = read_pickle(path)
                executable = deserialize_and_load(*serialized)
                abi.validate_full_state_abi(jax, serialized, executable, arguments[name], component="daily")
            del serialized
            memory = memory_sizes(executable)
            live = memory.get("estimated_live_bytes")
            if not isinstance(live, (int, float)) or not np.isfinite(live) or not 0 <= live <= 24 * 2**30:
                raise MemoryError("planned memory unavailable or above24GiB")
            executables[name] = executable
            journal.report["cold_measurements"][name] = {"executable": str(path), "load_seconds": time.monotonic() - began, "compiled_memory": memory}
        baselines = {}
        def execute(name, label):
            if journal.report["complete_AD_calls_attempted"] >= 8:
                raise ValueError("daily occupation admits at most eight complete AD calls")
            journal.report["complete_AD_calls_attempted"] += 1
            began = time.monotonic()
            with journal.deadline(args.call_seconds, name + label), source.forbid_model_compile():
                result = executables[name](*arguments[name])
                jax.block_until_ready(result)
            seconds = time.monotonic() - began
            journal.report["complete_AD_calls_completed"] += 1
            host = jax.device_get(result)
            path = args.output.with_suffix("." + name + "." + label + ".result.pkl")
            owner.atomic_pickle(path, {"identity": journal.report["identities"][name], "architecture": name, "label": label, "result": host})
            journal.report.setdefault("raw_results", []).append({"architecture": name, "label": label,
                "artifact": str(path), "synchronized_forward_AD_seconds": seconds})
            journal.emit("raw_AD_preserved_before_validation", architecture=name, label=label, artifact=str(path))
            check = leaf.check_result(jax, host, trees[name], samples)
            count, leaves = pair.parameter_gate(jax, trees[name], name)
            # The leaf checker gates all defined row values and exact masks;
            # ordinary auxiliary finite checks retain their original strict rule.
            ordinary_forward = (host[0][0], {key: value for key, value in host[0][1].items() if key != "state_diagnostics"})
            valid = adjacent("probe_shared_union_micro4").validate_result(jax, (ordinary_forward, host[1][0]), trees[name], count=count, leaves=leaves)
            if not check["finite"] or not check["physical_hard_counts_zero"] or check["initial_state_gradient_nonzero"] <= 0:
                raise ValueError("finite/physical/meaningful state AD gate failed")
            valid.update(hard_counts=check["hard_counts"], initial_state_shape=[4, 3854], initial_state_finite=True,
                         initial_state_nonzero=check["initial_state_gradient_nonzero"], row_state_diagnostics=check)
            peak = devices[0].memory_stats().get("peak_bytes_in_use")
            if not isinstance(peak, (int, float)) or not np.isfinite(peak) or not 0 <= peak <= 24 * 2**30:
                raise MemoryError("observed process peak unavailable or above24GiB")
            row = {"architecture": name, "label": label, "artifact": str(path), "synchronized_forward_AD_seconds": seconds,
                   "validation": valid, "observed_peak_bytes": peak,
                   "observed_peak_scope": "cumulative process including candidate compilation and both saved graphs"}
            if name in baselines:
                comparison = compare_repeats(jax, host, read_pickle(baselines[name])["result"], context["head"].contract)
                row["same_architecture_repeat"] = comparison
                if not all(comparison[key]["passed"] for key in ("forward_and_gates", "all_parameter_gradients", "all_initial_state_gradients")):
                    journal.report["numerical_failures"].append({"architecture": name, "label": label, "comparison": comparison, "actual_full_result": str(path), "reference_full_result": str(baselines[name])})
                    owner.atomic_json(args.output.with_suffix("." + name + "." + label + ".strict-failure.json"), journal.report["numerical_failures"][-1])
                if not comparison["forward_and_gates"]["passed"] or not comparison["all_parameter_gradients"]["passed"]:
                    journal.report["results"].append(row)
                    journal.emit("non_state_repeat_failed")
                    raise ValueError("parameter/forward/mask repeat failed")
            else:
                baselines[name] = path
            journal.report["results"].append(row)
            journal.emit("complete_AD_preserved", architecture=name, label=label, seconds=seconds)
            return row
        for name in ("original", "candidate"):
            row = execute(name, "cold-first")
            journal.report["cold_measurements"][name]["first_AD_seconds"] = row["synchronized_forward_AD_seconds"]
        for index in range(1, 4):
            order = ("original", "candidate") if index % 2 else ("candidate", "original")
            rows = {name: execute(name, "pair" + str(index)) for name in order}
            journal.report["pairs"].append({"pair": index, "order": order, **{name + "_seconds": row["synchronized_forward_AD_seconds"] for name, row in rows.items()}})
        medians = {name: float(np.median([row[name + "_seconds"] for row in journal.report["pairs"]])) for name in ("original", "candidate")}
        journal.report.update(status="numerical_failed" if (journal.report["numerical_failures"] or journal.report["inherited_source_strict_state_failures"]) else "passed",
            current_pair_state_parity_passed=not journal.report["numerical_failures"],
            current_pair_state_failure_count=len(journal.report["numerical_failures"]),
            warm_medians=medians, original_over_candidate=medians["original"] / medians["candidate"], complete_AD_calls=8,
            original_source_strict_repeat_failures_preserved=original_view["state_repeat_failure_records"],
            cold_calls_excluded_from_ratio=True, same_workload_complete_update_speed_measured=False)
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
