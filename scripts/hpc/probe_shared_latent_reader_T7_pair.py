"""Corrected original/fresh latent-reader recurrent native AD on one V100.

Separate compile stages preserve each sole graph before checks. Matched stages
only deserialize, execute synchronized complete T7 AD, and preserve results.
T7 remains the default; T30 requires its separate exact manifest. No training.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import json
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import time

import numpy as np

try:
    from scripts.hpc.probe_shared_leaf_full_state_cuda import adjacent, read_pickle, install_turnover_overlay, Journal, sha, BASE_SHA, CORRECTED_SHA, FIELD_SHA
except ModuleNotFoundError:
    from probe_shared_leaf_full_state_cuda import adjacent, read_pickle, install_turnover_overlay, Journal, sha, BASE_SHA, CORRECTED_SHA, FIELD_SHA


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("compile-original", "compile-candidate", "matched"), required=True)
    p.add_argument("--horizon", type=int, choices=(7, 30), default=7)
    for name in ("manifest", "prepared-context", "parameters", "field-adapter", "backbone-adapter", "transformer-module", "latent-module", "turnover-overlay", "runtime-preflight", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--turnover-base-sha256", default=BASE_SHA)
    p.add_argument("--turnover-sha256", "--turnover-corrected-sha256", dest="turnover_sha256", default=CORRECTED_SHA)
    p.add_argument("--original-executable", type=Path)
    p.add_argument("--candidate-executable", type=Path)
    p.add_argument("--wall-seconds", type=float, default=7000)
    p.add_argument("--lower-seconds", type=float, default=900)
    p.add_argument("--backend-seconds", type=float, default=6000)
    p.add_argument("--call-seconds", type=float, default=180)
    p.add_argument("--pairs", type=int, choices=(3, 4, 5), default=3)
    p.add_argument("--force-compilation-parallelism", type=int, choices=(0, 4), default=4,
        help="identical audited LLVM module parallelism for both graphs; 0 retains backend default partition count")
    p.add_argument("--validate-only", action="store_true", help="CPU-only exact asset/provenance validation; no model graph or CUDA")
    return p


@contextmanager
def candidate_backbone_patches(shared_process_head, expected_raw, backbone_assembler):
    """Retain the exact staged candidate binding; temporarily reuse its backbone."""
    original_raw = shared_process_head.shared_transformer_raw_outputs
    original_assembler = shared_process_head.assemble_shared_transformer_context
    if original_raw is not expected_raw:
        raise ValueError("production process-head raw binding differs from exact staged candidate")
    if not callable(expected_raw) or not callable(backbone_assembler):
        raise ValueError("candidate raw function and source-reuse assembler must be callable")
    if original_assembler is backbone_assembler:
        raise ValueError("diagnostic backbone assembler already installed in production process-head module")
    try:
        shared_process_head.assemble_shared_transformer_context = backbone_assembler
        yield
    finally:
        shared_process_head.shared_transformer_raw_outputs = original_raw
        shared_process_head.assemble_shared_transformer_context = original_assembler


def read_manifest(args):
    """Validate this T7 grant's protocol, independently of legacy schedules."""
    manifest = json.loads(args.manifest.read_text())
    horizon = getattr(args, "horizon", 7)
    schema = ("shared_corrected_latent_reader_T7_pair_manifest_v1" if horizon == 7
              else "shared_corrected_latent_reader_recurrent_pair_manifest_v1")
    if manifest.get("schema_version") != schema:
        raise ValueError("explicit corrected recurrent pair manifest/schema required")
    for name in ("source_profile", "input_bundle", "original_checkpoint", "candidate_parameters", "historical_original_admission"):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise ValueError("T7 manifest path required: "+name)
        if not Path(manifest[name]).is_file(): raise FileNotFoundError(manifest[name])
    expected = {"prepared_artifact_index": 0, "horizon": horizon, "native_AD_argnums": [0, 1],
        "precision": "FP32_neural_FP64_physics", "turnover_base_sha256": BASE_SHA, "turnover_sha256": CORRECTED_SHA,
        "training_updates": 0, "historical_original_is_corrected_pair_reference": False}
    if horizon == 7:
        expected.update(paid_jobs=False, T30_graph=False, independent_gpu_seconds=14400)
    else:
        expected.update(T30_graph=True, prepared_artifact_id=742,
                        execution_authorization_is_external=True)
    for name, value in expected.items():
        actual = manifest.get(name)
        if type(actual) is not type(value) or actual != value:
            raise ValueError("declared T7 protocol differs: "+name)
    if Path(manifest["candidate_parameters"]) != args.parameters:
        raise ValueError("manifest candidate initialization path differs from CLI")
    return manifest


def tree_digest(jax, tree):
    import hashlib
    digest = hashlib.sha256(str(jax.tree_util.tree_structure(tree)).encode())
    for value in jax.tree_util.tree_leaves(tree):
        value = np.ascontiguousarray(value)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def parameter_gate(jax, parameters, architecture):
    count, leaves = ((64884481, 300) if architecture == "original" else (66123713, 355))
    values = jax.tree_util.tree_leaves(parameters)
    if len(values) != leaves or sum(np.size(x) for x in values) != count:
        raise ValueError("exact full "+architecture+" parameter tree differs")
    if any(np.asarray(x).dtype != np.dtype("float32") or not np.all(np.isfinite(x)) for x in values):
        raise ValueError("full FP32 finite parameters required")
    return count, leaves


def validate_result(jax, host, parameters, architecture, initial_state):
    count, leaves = parameter_gate(jax, parameters, architecture)
    gates = adjacent("probe_shared_union_micro4")
    valid = gates.validate_result(jax, (host[0], host[1][0]), parameters, count=count, leaves=leaves)
    state = np.asarray(host[1][1])
    if state.shape != (3854,) or state.dtype != np.asarray(initial_state).dtype or not np.all(np.isfinite(state)) or not np.any(state != 0):
        raise gates.NumericalFailure("T7 initial state gradient nonfinite/zero/ABI differs")
    return {**valid, "initial_state_shape": [3854], "initial_state_finite": True,
        "initial_state_nonzero": int(np.count_nonzero(state)), "initial_state_coordinate2243": float(state[2243])}


def validation_failure_evidence(jax, host):
    """Bounded metadata plus every nonfinite coordinate; complete arrays stay saved."""
    (loss, auxiliary), (parameters, state) = host
    records = []
    for (path, value) in jax.tree_util.tree_flatten_with_path(host)[0]:
        a = np.asarray(value)
        finite = np.isfinite(a)
        if not np.all(finite):
            records.append({"path": jax.tree_util.keystr(path), "shape": list(a.shape),
                "nonfinite": [{"index": [int(i) for i in index], "value": str(a[tuple(index)])} for index in np.argwhere(~finite)],
                "finite_minimum": float(np.min(a[finite])) if np.any(finite) else None,
                "finite_maximum": float(np.max(a[finite])) if np.any(finite) else None})
    return {"loss": str(np.asarray(loss)), "hard_counts": {str(key): np.asarray(value).tolist() for key, value in auxiliary.get("hard", {}).items()},
        "nonfinite_leaves": records,
        "parameter_gradient_nonzero": sum(int(np.count_nonzero(x)) for x in jax.tree_util.tree_leaves(parameters)),
        "initial_state_gradient_nonzero": int(np.count_nonzero(state))}


def repeat_comparison(jax, actual, reference, contract):
    helper = adjacent("inspect_shared_leaf_cuda_acceptance")
    comparison = {"forward_and_gates": helper.compare_tree(jax, actual[0], reference[0]),
        "all_parameter_gradients": helper.compare_tree(jax, actual[1][0], reference[1][0]),
        "all_initial_state_gradients": helper.compare_tree(jax, actual[1][1], reference[1][1])}
    comparison["forward_failed_coordinate_ranges"] = failed_coordinate_ranges(jax, actual[0], reference[0])
    comparison["parameter_failed_coordinate_ranges"] = failed_coordinate_ranges(jax, actual[1][0], reference[1][0])
    a, r = np.asarray(actual[1][1]), np.asarray(reference[1][1])
    comparison["state_failure_coordinates"] = [{"coordinate": int(index),
        "bindings": helper.coordinate_binding(contract, int(index)), "actual": float(a[index]), "reference": float(r[index]),
        "absolute_difference": float(abs(a[index]-r[index])), "allowance": float(1e-6+1e-5*abs(r[index]))}
        for index in np.flatnonzero(~np.isclose(a, r, rtol=1e-5, atol=1e-6))]
    doc = []
    for leaf in contract.state_leaves:
        if "doc" in leaf.key.lower() and leaf.start is not None and leaf.stop is not None:
            aa, rr = a[leaf.start:leaf.stop], r[leaf.start:leaf.stop]
            doc.append({"field": leaf.key, "start": leaf.start, "stop": leaf.stop,
                "failed": int(np.count_nonzero(~np.isclose(aa, rr, rtol=1e-5, atol=1e-6))),
                "maximum_absolute_difference": float(np.max(np.abs(aa-rr))) if aa.size else 0.0})
    comparison["DOC_state_gradients"] = doc
    comparison["passed"] = all(comparison[name]["passed"] for name in ("forward_and_gates", "all_parameter_gradients", "all_initial_state_gradients"))
    return comparison


def failed_coordinate_ranges(jax, actual, reference):
    """Complete compact failure coverage in C-order, with value/error bounds.

    Every coordinate in each half-open range fails the unchanged tolerance.
    Exact per-coordinate values remain in the linked complete result pickles.
    """
    if jax.tree_util.tree_structure(actual) != jax.tree_util.tree_structure(reference):
        raise ValueError("failed-coordinate tree differs")
    records = []
    for (path, actual), reference in zip(jax.tree_util.tree_flatten_with_path(actual)[0], jax.tree_util.tree_leaves(reference), strict=True):
        a, r = np.asarray(actual), np.asarray(reference)
        if a.shape != r.shape or a.dtype != r.dtype:
            raise ValueError("failed-coordinate leaf ABI differs")
        indices = np.flatnonzero(~np.isclose(a, r, rtol=1e-5, atol=1e-6))
        if not indices.size: continue
        ranges = []
        for group in np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1):
            start, stop = int(group[0]), int(group[-1])+1
            aa, rr = a.reshape(-1)[start:stop].astype(np.float64), r.reshape(-1)[start:stop].astype(np.float64)
            error = aa-rr
            def bounds(values):
                finite = np.isfinite(values)
                return {"minimum": float(np.min(values[finite])) if np.any(finite) else None,
                    "maximum": float(np.max(values[finite])) if np.any(finite) else None,
                    "nonfinite": int(np.count_nonzero(~finite))}
            ranges.append({"flat_start": start, "flat_stop_exclusive": stop, "failed_coordinates": stop-start,
                "actual": bounds(aa), "reference": bounds(rr), "signed_error": bounds(error),
                "absolute_error": bounds(np.abs(error)), "allowance": bounds(1e-6+1e-5*np.abs(rr))})
        records.append({"path": jax.tree_util.keystr(path), "shape": list(a.shape), "dtype": str(a.dtype),
            "failed_coordinates": int(indices.size), "C_order_half_open_ranges": ranges})
    return {"complete": True, "failed_coordinates": sum(row["failed_coordinates"] for row in records), "leaves": records}


def assets(jax, args):
    """Reuse exact admitted original inputs and independently saved fresh tree."""
    source = adjacent("probe_shared_original_full_state_schedule")
    loader = adjacent("probe_shared_latent_reader")
    package = "research.daily_coarse_graining.daily_operator."
    transformer = loader.load(args.transformer_module, package+"shared_transformer")
    latent = loader.load(args.latent_module, package+"shared_latent_reader")
    source.load_overlays()
    owner = adjacent("probe_shared_expert_real_daily")
    manifest = read_manifest(args)
    horizon = getattr(args, "horizon", 7)
    component = f"T{horizon}"
    profile, reference = source.read_reference(manifest)
    historical = json.loads(Path(manifest["historical_original_admission"]).read_text())
    # Historical admission supplies input/source provenance only. Its old
    # uncorrected result is never admitted as the new corrected pair baseline.
    adjacent("shared_full_state_ad_execution").validate_original_admission(historical, component, reference)
    root = Path(os.environ.get("ORCHIDEE_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != reference["source_git_head"]: raise ValueError("immutable source head differs")
    context = read_pickle(args.prepared_context)
    expected_context = owner.context_identity(reference, Path(manifest["original_checkpoint"]), reference["method"]["training_plan"]["training_artifact_ids"][0])
    if context.get("identity") != expected_context or Path(context["daily_executable"]).name != Path(profile["executables"]["daily"]["executable"]).name:
        raise ValueError("prepared context provenance differs")
    bundle = read_pickle(manifest["input_bundle"])
    source.validate_bundle(jax, bundle, manifest, reference)
    case = bundle["payloads"][0]["cases"][horizon]
    if (len(case) != 5 or np.shape(case[0]) != (3854,)
            or np.shape(case.days[0].continuous_state)[0] != horizon
            or (horizon == 30 and int(bundle["artifact_ids"][0]) != 742)):
        raise ValueError("exact public native recurrent case required")
    config = context["head"].shared_config
    if (config.width != 768 or config.backbone_blocks != 8 or config.coordinate_blocks != 2
            or config.output_chunk_size != 64 or config.decoder_execution != "gather_dynamic_reuse"
            or getattr(config, "decoder_architecture", "coordinate") != "coordinate"):
        raise ValueError("original architecture differs")
    sources = {}
    for key, path, historical_key in (("field_expert_adapter", args.field_adapter, "candidate_decoder_sha256"),
                                     ("backbone_adapter", args.backbone_adapter, "candidate_backbone_sha256")):
        digest = sha(path)
        if digest != historical["identity"][historical_key] or (key == "field_expert_adapter" and digest != FIELD_SHA):
            raise ValueError("historical original neural source differs: "+key)
        sources[key] = {"path": str(path), "sha256": digest}
    siblings = [args.field_adapter.with_name(name+".py") for name in ("shared_decoder_dense", "shared_decoder_macro_packed", "shared_decoder_expert_execution")]
    siblings += [args.backbone_adapter.with_name(name+".py") for name in ("shared_backbone_projection_reuse", "shared_backbone_execution")]
    sources["siblings"] = {str(path): sha(path) for path in siblings}
    if args.turnover_base_sha256 != BASE_SHA or args.turnover_sha256 != CORRECTED_SHA or sha(args.turnover_overlay) != CORRECTED_SHA:
        raise ValueError("exact accepted guarded turnover overlay required")
    candidate_head = replace(context["head"], shared_config=latent.shared_latent_reader_config(config))
    candidate_config = candidate_head.shared_config
    if candidate_config.latent_reader_execution != "field" or candidate_config.output_chunk_size != 64:
        raise ValueError("canonical field/chunk64 candidate required")
    if (latent.LATENT_SLOTS, latent.LATENT_LAYERS, latent.READER_WIDTH, latent.READER_LAYERS) != (32, 2, 64, 2):
        raise ValueError("fixed latent32/private-reader64x2 prototype differs")
    seed = int(reference["method"]["daily_protocol"]["seed"])
    candidate_identity = {"source_head": head, "artifact_id": int(bundle["artifact_ids"][0]), "input_source_method": reference["method"],
        "seed": seed, "candidate_config": vars(candidate_config),
        "candidate_source_sha256": {"shared_transformer.py": sha(args.transformer_module), "shared_latent_reader.py": sha(args.latent_module)},
        "source_reuse_assembler_sha256": sha(args.backbone_adapter),
        "source_reuse_projection_sha256": sha(args.backbone_adapter.with_name("shared_backbone_projection_reuse.py"))}
    saved = read_pickle(args.parameters)
    if saved.get("identity") != candidate_identity:
        raise ValueError("exact saved fresh candidate initialization identity differs")
    candidate_parameters = saved["parameters"]
    parameter_gate(jax, candidate_parameters, "candidate")
    del saved
    from scripts.hpc.run_shared_learning import load_learning_checkpoint
    original = load_learning_checkpoint(Path(manifest["original_checkpoint"]), expected_method=reference["method"])
    if original.get("completed_updates") != 0: raise ValueError("original diagnostic update0 required")
    original_parameters = original["parameters"]
    parameter_gate(jax, original_parameters, "original")
    del original
    compiler_options = {"xla_gpu_enable_llvm_module_compilation_parallelism": True}
    if args.force_compilation_parallelism:
        compiler_options["xla_gpu_force_compilation_parallelism"] = args.force_compilation_parallelism
    from research.daily_coarse_graining.daily_operator import shared_recurrent_training
    scientific_functions = {}
    for name in ("make_shared_recurrent_objective", "make_canonical_step", "make_rollout_loss"):
        function = getattr(shared_recurrent_training, name)
        text = inspect.getsource(function).replace("\r\n", "\n")
        scientific_functions[name] = {"sha256": hashlib.sha256(text.encode()).hexdigest(),
            "source_file": inspect.getsourcefile(function), "source_line": inspect.getsourcelines(function)[1]}
    common = {"source_git_head": head, "manifest_sha256": sha(args.manifest), "input_bundle_sha256": sha(manifest["input_bundle"]),
        "prepared_context_sha256": sha(args.prepared_context),
        "compiler_options": compiler_options,
        "neural_module_sha256": {"shared_transformer.py": sha(args.transformer_module), "shared_latent_reader.py": sha(args.latent_module)},
        "scientific_functions": scientific_functions,
        "historical_source_admission": {"path": manifest["historical_original_admission"], "sha256": sha(manifest["historical_original_admission"]), "corrected_pair_reference": False},
        "input_case_sha256": tree_digest(jax, case), "artifact_id": int(bundle["artifact_ids"][0]), "horizon": horizon,
        "method": reference["method"], "prepared_context_identity": expected_context, "sources": sources,
        "turnover_base_sha256": BASE_SHA, "turnover_sha256": CORRECTED_SHA,
        "native_AD_argnums": [0, 1], "precision": "FP32_neural_FP64_physics", "training_updates": 0,
        "jax_version": jax.__version__}
    identities = {
        "original": {**common, "architecture": "original", "parameter_count": 64884481, "parameter_leaves": 300,
            "actual_execution": {"decoder_execution": "expert128", "decoder_chunk_size": 128,
                "GPP_decoder": "canonical", "GPP_chunk_size": 64, "dynamic_projection_execution": "field",
                "backbone_execution": "source_reuse"},
            "parameter_provenance": {"checkpoint": manifest["original_checkpoint"], "sha256": sha(manifest["original_checkpoint"]), "completed_updates": 0}},
        "candidate": {**common, "architecture": "shared_latent32x384_reader64x2", "parameter_count": 66123713, "parameter_leaves": 355,
            "actual_execution": {"decoder_execution": "field", "decoder_chunk_size": 64,
                "backbone_execution": "source_reuse"},
            "parameter_provenance": {"artifact": str(args.parameters), "sha256": sha(args.parameters), "initialization": "fresh_public_whole_tree", "identity": candidate_identity}}}
    identities = json.loads(json.dumps(identities))
    return source, owner, transformer, context, candidate_head, case, {"original": original_parameters, "candidate": candidate_parameters}, identities


def run(args):
    component = f"T{getattr(args, 'horizon', 7)}"
    if not 1 <= args.wall_seconds <= 7200 or min(args.lower_seconds, args.backend_seconds, args.call_seconds) <= 0:
        raise ValueError("session wall≤7200 and positive deadlines required")
    if not args.validate_only and args.stage == "matched" and not (args.original_executable and args.candidate_executable):
        raise ValueError("matched execution requires both saved graphs; no compilation fallback")
    if args.validate_only:
        os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or "," in visible or visible.strip() != visible: raise ValueError("one parent allocated GPU required")
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adjacent("training_artifact_io")
    journal = Journal(args.output, args.wall_seconds)
    journal.report.update(schema_version=("shared_corrected_original_latent_reader_T7_pair_v1" if component == "T7" else "shared_corrected_original_latent_reader_recurrent_pair_v1"), requested_stage=args.stage,
        component=component, training_updates=0, gpu_count=1, precision="FP32_neural_FP64_physics",
        native_AD_argnums=[0, 1], cross_architecture_parity_is_gate=False, model_compiles={"original": 0, "candidate": 0},
        cold_measurements={}, pairs=[], numerical_failures=[])
    journal.report.update(runner_harness_sha256=sha(Path(__file__)),
        executable_identity_scope="mathematical_graph_sources_compiler_ABI_inputs_and_parameters; runner_logging_SHA_recorded_separately")
    try:
        if not args.validate_only: adjacent("orcjax_gpu_runtime").initialize_gpu_backend()
        import jax
        jax.config.update("jax_enable_x64", True)
        devices = jax.devices()
        if not args.validate_only and (len(devices) != 1 or devices[0].platform != "gpu" or "V100" not in devices[0].device_kind):
            raise ValueError("exactly one allocated free-node V100 required")
        if not args.validate_only:
            journal.report["runtime_preflight"] = adjacent("probe_shared_union_micro4").validate_runtime_report(args.runtime_preflight, jax.__version__)
        journal.emit("loading_exact_"+component+"_assets")
        source, owner, transformer, context, candidate_head, case, parameters, identities = assets(jax, args)
        for identity in identities.values(): identity["device_kind"] = devices[0].device_kind
        journal.report["identities"] = identities
        journal.report["actual_execution"] = {architecture: identity["actual_execution"] for architecture, identity in identities.items()}
        journal.report["compiler_options"] = identities["original"]["compiler_options"]
        if args.validate_only:
            if jax.default_backend() != "cpu": raise ValueError("validate-only requires CPU backend")
            journal.report.update(status="assets_validated", GPU_executed=False, gpu_count=0, model_compiles={"original": 0, "candidate": 0},
                parameter_gates={architecture: dict(zip(("count", "leaves"), parameter_gate(jax, tree, architecture))) for architecture, tree in parameters.items()})
            journal.emit("CPU_assets_validated_no_graph")
            return journal.report
        from jax.experimental.serialize_executable import serialize, deserialize_and_load
        from scripts.hpc.run_shared_transformer_admission import memory_sizes
        abi = adjacent("shared_full_state_ad_execution")
        device_case = jax.device_put(case)
        architectures = ("original", "candidate") if args.stage == "matched" else (args.stage.removeprefix("compile-"),)
        executables, arguments, baseline_paths = {}, {}, {}

        def execute(architecture, label, *, pair=None):
            journal.emit(""+component+"_call_entered", architecture=architecture, label=label)
            began = time.monotonic()
            with journal.deadline(args.call_seconds, "call_"+architecture+"_"+label), source.forbid_model_compile():
                result = executables[architecture](*arguments[architecture])
                jax.block_until_ready(result)
            synchronized_seconds = time.monotonic()-began
            transfer_started = time.monotonic()
            host = jax.device_get(result)
            path = args.output.with_suffix("."+architecture+"."+label+".result.pkl")
            if path.exists(): raise FileExistsError(path)
            owner.atomic_pickle(path, {"identity": identities[architecture], "architecture": architecture, "label": label, "result": host})
            del result
            transfer_preserve_seconds = time.monotonic()-transfer_started
            raw_record = {"architecture": architecture, "label": label, "artifact": str(path),
                "synchronized_forward_AD_seconds": synchronized_seconds, "transfer_and_preserve_seconds": transfer_preserve_seconds}
            journal.report.setdefault("raw_results", []).append(raw_record)
            journal.emit(""+component+"_raw_result_preserved", **raw_record)
            try:
                valid = validate_result(jax, host, parameters[architecture], architecture, case[0])
            except Exception as error:
                evidence = validation_failure_evidence(jax, host)
                journal.report.setdefault("validation_failures", []).append({**raw_record, "error_type": type(error).__name__, "error": str(error), **evidence})
                journal.emit(""+component+"_preserved_result_validation_failed", **raw_record, error_type=type(error).__name__, error=str(error))
                raise
            stats = devices[0].memory_stats()
            peak = stats.get("peak_bytes_in_use") if isinstance(stats, dict) else None
            row = {"architecture": architecture, "label": label, "artifact": str(path), "synchronized_forward_AD_seconds": synchronized_seconds,
                "transfer_and_preserve_seconds": transfer_preserve_seconds, "validation": valid, "JAX_device_memory_stats": stats,
                "observed_peak_bytes": peak, "observed_peak_memory_passed": isinstance(peak, (int, float)) and np.isfinite(peak) and 0 <= peak < 24*2**30}
            if architecture in baseline_paths:
                baseline = read_pickle(baseline_paths[architecture])
                row["same_architecture_repeat"] = repeat_comparison(jax, host, baseline["result"], context["head"].contract)
                row["same_architecture_repeat"]["reference_artifact"] = str(baseline_paths[architecture])
                if not row["same_architecture_repeat"]["passed"]:
                    failure_path = args.output.with_suffix("."+architecture+"."+label+".strict-failure-ranges.json")
                    owner.atomic_json(failure_path, {"identity": identities[architecture], "architecture": architecture, "label": label,
                        "actual_full_result": str(path), "reference_full_result": str(baseline_paths[architecture]),
                        "rtol": 1e-5, "atol": 1e-6, "comparison": row["same_architecture_repeat"]})
                    row["strict_failure_ranges_artifact"] = str(failure_path)
                    journal.report["numerical_failures"].append({"architecture": architecture, "label": label, "artifact": str(path), "comparison": row["same_architecture_repeat"]})
                del baseline
            else:
                baseline_paths[architecture] = path
            journal.report["results"].append(row)
            del host
            journal.emit(""+component+"_complete_result_preserved", **row)
            if peak is not None and peak >= 24*2**30: raise MemoryError("observed JAX peak not below24GiB")
            return row

        for architecture in architectures:
            arguments[architecture] = (jax.device_put(parameters[architecture]), *device_case)
            existing = getattr(args, architecture+"_executable")
            path = existing or args.output.with_suffix(".executable.pkl")
            cold = {}
            if existing:
                sidecar = json.loads(path.with_suffix(path.suffix+".identity.json").read_text())
                if sidecar != {"identity": identities[architecture], "component": component}: raise ValueError("saved corrected T7 exact identity differs")
                began = time.monotonic()
                with source.forbid_model_compile():
                    serialized = read_pickle(path)
                    executable = deserialize_and_load(*serialized)
                    abi.validate_full_state_abi(jax, serialized, executable, arguments[architecture], component=component)
                cold["load_seconds"] = time.monotonic()-began
                journal.emit("saved_corrected_"+component+"_loaded", architecture=architecture, executable=str(path), load_seconds=cold["load_seconds"])
            else:
                if path.exists() or path.with_suffix(path.suffix+".identity.json").exists():
                    raise FileExistsError("preserved graph exists; use explicit saved-executable recovery: "+str(path))
                from research.daily_coarse_graining.daily_operator import shared_process_head
                from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_recurrent_objective
                from jax_orchidee.stomate import carbon_kernels
                with ExitStack() as patches:
                    patches.enter_context(install_turnover_overlay(carbon_kernels, args.turnover_overlay, BASE_SHA))
                    backbone_assembler = owner.load_backbone_assembler(args.backbone_adapter)
                    if architecture == "original":
                        adapter = owner.load_staged_adapter(args.field_adapter)
                        patches.enter_context(owner.candidate_process_head_patches(shared_process_head, adapter, backbone_assembler))
                    else:
                        patches.enter_context(candidate_backbone_patches(shared_process_head,
                            transformer.shared_transformer_raw_outputs, backbone_assembler))
                    selected_head = context["head"] if architecture == "original" else candidate_head
                    _, objective = make_shared_recurrent_objective(selected_head, context["template"], context["state_scales"], context["science_scales"])
                    backward = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True),
                        compiler_options=identities[architecture]["compiler_options"])
                    journal.emit(""+component+"_lowering", architecture=architecture)
                    began = time.monotonic()
                    with journal.deadline(args.lower_seconds, "lower_"+architecture): lowered = backward.lower(*arguments[architecture])
                    cold["lower_seconds"] = time.monotonic()-began
                    journal.emit(""+component+"_backend_compiling", architecture=architecture, lower_seconds=cold["lower_seconds"])
                    began = time.monotonic()
                    journal.report["model_compiles"][architecture] = 1
                    with journal.deadline(args.backend_seconds, "backend_"+architecture): executable = lowered.compile()
                    cold["compile_seconds"] = time.monotonic()-began
                    # Preserve this sole expensive graph before any validation.
                    serialized = serialize(executable)
                    if path.exists(): raise FileExistsError(path)
                    owner.atomic_pickle(path, serialized)
                    owner.atomic_json(path.with_suffix(path.suffix+".identity.json"), {"identity": identities[architecture], "component": component})
                    journal.emit("corrected_"+component+"_serialized", architecture=architecture, executable=str(path), compile_seconds=cold["compile_seconds"])
                    abi.validate_full_state_abi(jax, serialized, executable, arguments[architecture], component=component)
            del serialized
            memory = memory_sizes(executable)
            cold.update(executable=str(path), compiled_memory=memory)
            # Metadata from this already saved/loaded executable only. This
            # optional observation neither compiles nor changes its graph.
            cost_started = time.monotonic()
            try:
                with source.forbid_model_compile():
                    cost = executable.cost_analysis()
                cold["cost_analysis"] = {"available": True, "value": json.loads(json.dumps(cost, default=str))}
            except Exception as error:
                cold["cost_analysis"] = {"available": False, "error_type": type(error).__name__, "error": str(error)}
            cold["cost_analysis_seconds"] = time.monotonic()-cost_started
            journal.report["cold_measurements"][architecture] = cold
            if memory["estimated_live_bytes"] >= 24*2**30: raise MemoryError("compiled T7 live memory not below24GiB")
            executables[architecture] = executable
            cold["first_call"] = execute(architecture, "cold-first")
        if args.stage == "matched":
            for pair in range(1, args.pairs+1):
                order = ("original", "candidate") if pair % 2 else ("candidate", "original")
                rows = [execute(architecture, f"pair{pair}", pair=pair) for architecture in order]
                timing = {row["architecture"]: row["synchronized_forward_AD_seconds"] for row in rows}
                journal.report["pairs"].append({"pair": pair, "order": list(order), "measurements": rows,
                    "candidate_speedup": timing["original"]/timing["candidate"]})
                journal.emit("matched_pair_preserved", pair=pair, measurements=timing)
        journal.report.update(status="numerical_failed" if journal.report["numerical_failures"] else "passed",
            completed_recurrent_calls=len(journal.report["results"]), observed_peak_memory_passed=all(x["observed_peak_memory_passed"] for x in journal.report["results"]))
        if component == "T7": journal.report["completed_T7_calls"] = len(journal.report["results"])
        if journal.report["pairs"]:
            timing = {architecture: [next(row["synchronized_forward_AD_seconds"] for row in pair["measurements"] if row["architecture"] == architecture) for pair in journal.report["pairs"]] for architecture in architectures}
            journal.report["matched_timing_summary"] = {"synchronized_seconds": timing,
                "original_median_seconds": float(np.median(timing["original"])), "candidate_median_seconds": float(np.median(timing["candidate"])),
                "median_ratio_speedup": float(np.median(timing["original"])/np.median(timing["candidate"]))}
        journal.emit("completed")
        return journal.report
    except Exception as error:
        journal.report.update(status="resource_failed" if isinstance(error, MemoryError) else "error", error_type=type(error).__name__, error=str(error))
        journal.emit("error")
        raise
    finally:
        journal.timer.cancel()


if __name__ == "__main__":
    run(parser().parse_args())
