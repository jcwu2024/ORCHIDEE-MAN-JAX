"""One corrected original micro4 graph, native parameter and initial-state AD.

Diagnostic only: no updates, architecture changes, cached neural output replay,
or scientific adoption. The physical overlay carries its audited Fortran
provenance; the scalar public loss and all physical gates remain unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np

# Loading this helper does not select a JAX backend.
try:
    from scripts.hpc.probe_shared_biomass_state_amplification import adjacent, read_pickle, validate_snapshot, select_actual_sample, install_turnover_overlay
except ModuleNotFoundError:
    from probe_shared_biomass_state_amplification import adjacent, read_pickle, validate_snapshot, select_actual_sample, install_turnover_overlay

SCHEMA = "shared_leaf_corrected_full_state_cuda_v1"
BASE_SHA = "bb8674f3d2ceadb28df907346d18f6300dd87f46d09ff6010c795f72dbee6485"
CORRECTED_SHA = "f6240cf5ce64a47b9bbe0647b5e5715305e9a9230ea77f2bd9f42981ae35a93c"
FIELD_SHA = "5d77141bad1359d3a71174ba03eba57527d7b0e94cf3e42bce73a2dc650c740d"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def retain_row_diagnostics(module):
    """Observation only: keep state arrays/masks; preserve all original reducers."""
    original = module._aggregate_auxiliary
    def aggregate(jax, jnp, auxiliary):
        reduced = original(jax, jnp, {k: v for k, v in auxiliary.items() if k != "state_diagnostics"})
        reduced["state_diagnostics"] = auxiliary["state_diagnostics"]
        return reduced
    module._aggregate_auxiliary = aggregate
    try:
        yield
    finally:
        module._aggregate_auxiliary = original


def resource_snapshot():
    result = {"pid": os.getpid(), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    for name in ("sched_getaffinity", "getloadavg"):
        if hasattr(os, name):
            result[name] = list(getattr(os, name)(0) if name == "sched_getaffinity" else getattr(os, name)())
    try:
        result["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        result["nvidia_smi_error"] = str(error)
    return result


class Journal:
    def __init__(self, output, wall_seconds):
        self.output, self.started = output, time.monotonic()
        self.report = {"schema_version": SCHEMA, "status": "running", "scientific_acceptance": False,
                       "adoption": False, "stage_events": [], "results": [], "resources": resource_snapshot()}
        self.timer = threading.Timer(wall_seconds, lambda: self.expire("wall"))
        self.timer.daemon = True
        self.timer.start()

    def emit(self, stage, **values):
        from scripts.hpc.training_artifact_io import _atomic_json
        self.report.update(stage=stage, elapsed_seconds=time.monotonic()-self.started)
        self.report["stage_events"].append({"stage": stage, "elapsed_seconds": self.report["elapsed_seconds"], **values})
        _atomic_json(self.output, self.report)
        print(json.dumps({"stage": stage, **values}, default=str), flush=True)

    def expire(self, phase):
        self.report.update(status="incomplete_timeout", deadline_phase=phase)
        try:
            self.emit("deadline_exceeded", phase=phase)
        finally:
            os._exit(124)

    @contextmanager
    def deadline(self, seconds, phase):
        timer = threading.Timer(seconds, lambda: self.expire(phase))
        timer.daemon = True
        timer.start()
        try:
            yield
        finally:
            timer.cancel()


def prepare(jax, args):
    """Shared source/input validation for CUDA and independent CPU JVP."""
    source = adjacent("probe_shared_original_full_state_schedule")
    adjacent("shared_full_state_ad_execution")
    owner = adjacent("probe_shared_expert_real_daily")
    manifest = source.read_manifest(args.manifest)
    if manifest.get("resume", {}).get("completed_pairs") != 1:
        raise ValueError("exact independent saved step1 chains required")
    for schedule in ("old", "new"):
        if str(getattr(args, schedule+"_snapshot")) != manifest["resume"][schedule]:
            raise ValueError("snapshot path differs from explicit manifest resume")
    profile, reference = source.read_reference(manifest)
    source.validate_resume_metadata(manifest, reference)
    source.validate_original_assets(manifest, reference)
    root = Path(os.environ.get("ORCHIDEE_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != reference["source_git_head"]:
        raise ValueError("immutable checkout head differs")
    context = read_pickle(args.prepared_context)
    expected = owner.context_identity(reference, Path(manifest["original_checkpoint"]), reference["method"]["training_plan"]["training_artifact_ids"][0])
    if context.get("identity") != expected or Path(context["daily_executable"]).name != Path(profile["executables"]["daily"]["executable"]).name:
        raise ValueError("prepared context provenance differs")
    config = context["head"].shared_config
    if (config.width != 768 or config.backbone_blocks != 8 or config.coordinate_blocks != 2
            # Immutable c599 predates these selectors and contains only the
            # original coordinate architecture. Its historical field adapter
            # is fixed by the exact source hash below.
            or getattr(config, "decoder_architecture", "coordinate") != "coordinate"
            or getattr(config, "latent_reader_execution", "field") != "field"):
        raise ValueError("original complete architecture required")
    bundle = read_pickle(manifest["input_bundle"])
    workload = source.validate_bundle(jax, bundle, manifest, reference)
    _, binding = select_actual_sample(jax, bundle, 275, 17)
    if binding["micro4_sha256"] != workload["daily_micro4_sha256"][binding["artifact_index"]][4]:
        raise ValueError("actual artifact275 rows16:20 binding differs")
    daily = bundle["payloads"][binding["artifact_index"]]["daily"]
    samples = jax.tree_util.tree_map(lambda value: np.asarray(value)[16:20], daily)
    admission = json.loads(Path(manifest["components"]["daily"]["report"]).read_text())
    sources = {}
    for key, path in (("field_expert_adapter", args.field_adapter), ("backbone_adapter", args.backbone_adapter)):
        digest, historic = sha(path), admission["source_artifacts"][key]["sha256"]
        if digest != historic or (key == "field_expert_adapter" and digest != FIELD_SHA):
            raise ValueError("exact historical original source artifact required: "+key)
        sources[key] = {"path": str(path), "sha256": digest, "historic_sha256": historic}
    # Bind every staged sibling, not only entry points, into recovery identity.
    siblings = [args.field_adapter.with_name(stem+".py") for stem in ("shared_decoder_dense", "shared_decoder_macro_packed", "shared_decoder_expert_execution")]
    siblings += [args.backbone_adapter.with_name(stem+".py") for stem in ("shared_backbone_projection_reuse", "shared_backbone_execution")]
    sources["siblings"] = {str(path): sha(path) for path in siblings}
    if args.turnover_base_sha256 != BASE_SHA or args.turnover_sha256 != CORRECTED_SHA or sha(args.turnover_overlay) != CORRECTED_SHA:
        raise ValueError("accepted exact normalized turnover source overlay required")
    identity = {"source_git_head": head, "manifest_sha256": sha(args.manifest),
        "harness_sha256": sha(Path(__file__)),
        "AD_helper_sha256": {name: sha(Path(__file__).with_name(name+".py")) for name in ("shared_daily_dual_ad", "shared_daily_microbatch", "shared_full_state_ad_execution")},
        "prepared_context_identity": expected, "input_binding": binding, "sources": sources,
        "turnover_base_sha256": BASE_SHA, "turnover_sha256": CORRECTED_SHA,
        "snapshots": {schedule: {"path": str(getattr(args, schedule+"_snapshot")), "sha256": sha(getattr(args, schedule+"_snapshot"))} for schedule in ("old", "new")},
        "parameters": 64884481, "parameter_leaves": 300, "native_AD_argnums": [0, 1],
        "precision": "FP32_neural_FP64_physical", "loss": "unchanged_public_joint_daily_mean4",
        "state_diagnostics": "unreduced_four_rows_observation_only", "jax_version": jax.__version__}
    # Sidecars use JSON's representation; normalize tuples before recovery compare.
    identity = json.loads(json.dumps(identity))
    return source, owner, manifest, reference, context, samples, identity


def check_result(jax, result, parameters, samples):
    (loss, auxiliary), (parameter_gradient, state_gradient) = result
    if jax.tree_util.tree_structure(parameters) != jax.tree_util.tree_structure(parameter_gradient):
        raise ValueError("parameter gradient tree differs")
    gradients = jax.tree_util.tree_leaves(parameter_gradient)
    for p, g in zip(jax.tree_util.tree_leaves(parameters), gradients, strict=True):
        if np.shape(p) != np.shape(g) or np.asarray(g).dtype != np.dtype("float32"):
            raise ValueError("parameter gradient shape/dtype differs")
    state = np.asarray(state_gradient)
    if state.shape != (4, 3854) or state.dtype != np.asarray(samples.inputs.continuous_state).dtype:
        raise ValueError("state gradient ABI differs")
    ordinary = {k: v for k, v in auxiliary.items() if k != "state_diagnostics"}
    finite = all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves((loss, ordinary, parameter_gradient, state)))
    diagnostics = auxiliary["state_diagnostics"]
    state_shapes = {}
    # Public diagnostics are scientific groups, not the canonical flat AD input.
    # StateFeatureGroups and StateDefinedMasks intentionally have distinct types.
    for values_key, masks_key in (("predicted", "predicted_defined"), ("expected", "expected_defined")):
        values, masks = diagnostics[values_key], diagnostics[masks_key]
        if getattr(values, "_fields", None) != getattr(masks, "_fields", None):
            raise ValueError("state group/mask fields differ")
        value_leaves, mask_leaves = jax.tree_util.tree_leaves(values), jax.tree_util.tree_leaves(masks)
        if len(value_leaves) != len(mask_leaves):
            raise ValueError("state diagnostic mask leaf count differs")
        state_shapes[values_key] = []
        for value, mask in zip(value_leaves, mask_leaves, strict=True):
            value, mask = np.asarray(value), np.asarray(mask)
            if not value.shape or value.shape[0] != 4 or mask.shape != value.shape or mask.dtype != np.dtype("bool"):
                raise ValueError("row scientific state group/mask ABI differs")
            finite = finite and bool(np.all(np.isfinite(value[mask])))
            state_shapes[values_key].append(list(value.shape))
    for key, expected in (("expected", samples.expected_next_state), ("expected_defined", samples.expected_next_state_defined)):
        if jax.tree_util.tree_structure(diagnostics[key]) != jax.tree_util.tree_structure(expected):
            raise ValueError("state diagnostic expected group structure differs")
        for value, sample_value in zip(jax.tree_util.tree_leaves(diagnostics[key]), jax.tree_util.tree_leaves(expected), strict=True):
            if not np.array_equal(value, sample_value, equal_nan=True):
                raise ValueError("state diagnostic expected input changed")
    hard = adjacent("probe_shared_expert_real_daily").hard_counts(auxiliary)
    return {"loss": float(np.asarray(loss)), "finite": bool(finite), "hard_counts": hard,
            "physical_hard_counts_zero": not any(hard.values()),
            "parameter_gradient_nonzero": sum(int(np.count_nonzero(x)) for x in gradients),
            "initial_state_gradient_nonzero": int(np.count_nonzero(state)),
            "state_diagnostic_group_shapes": state_shapes,
            "coordinate2243": state[:, 2243].tolist()}


def compare_repeats(jax, first, current):
    """Report drift independently, including every parameter/state/aux leaf."""
    if jax.tree_util.tree_structure(first) != jax.tree_util.tree_structure(current):
        raise ValueError("repeat result tree differs")
    records = []
    for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(first)[0], jax.tree_util.tree_leaves(current), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        equal = bool(np.array_equal(a, b))
        if not equal:
            records.append({"path": jax.tree_util.keystr(path), "shape": list(a.shape),
                            "maximum_absolute_difference": float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))),
                            "strict_rtol1e5_atol1e6": bool(np.allclose(a, b, rtol=1e-5, atol=1e-6))})
    return {"bitwise_equal": not records, "strict_parity": all(x["strict_rtol1e5_atol1e6"] for x in records), "changed_leaves": records}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "prepared-context", "old-snapshot", "new-snapshot", "field-adapter", "backbone-adapter", "turnover-overlay", "runtime-preflight", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--turnover-base-sha256", default=BASE_SHA)
    p.add_argument("--turnover-sha256", "--turnover-corrected-sha256", dest="turnover_sha256", default=CORRECTED_SHA)
    p.add_argument("--saved-executable", type=Path)
    p.add_argument("--original-reference", type=Path, help="optional admitted immutable original diagnostic control")
    p.add_argument("--wall-seconds", type=float, default=6000)
    p.add_argument("--lower-seconds", type=float, default=900)
    p.add_argument("--backend-seconds", type=float, default=4200)
    p.add_argument("--call-seconds", type=float, default=180)
    p.add_argument("--repeats", type=int, choices=(2,), default=2)
    return p


def run(args):
    if not 1 <= args.wall_seconds <= 7200 or min(args.lower_seconds, args.backend_seconds, args.call_seconds) <= 0:
        raise ValueError("bounded positive deadlines required; wall at most7200s")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible.strip() != visible:
        raise ValueError("parent must explicitly allocate exactly one visible GPU")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    adjacent("training_artifact_io")
    journal = Journal(args.output, args.wall_seconds)
    try:
        journal.emit("gpu_backend_initializing")
        adjacent("orcjax_gpu_runtime").initialize_gpu_backend()
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu" or "V100" not in devices[0].device_kind:
            raise ValueError("exactly one allocated V100 required")
        preflight = adjacent("probe_shared_union_micro4").validate_runtime_report(args.runtime_preflight, jax.__version__)
        journal.emit("source_and_input_validation")
        source, owner, manifest, reference, context, samples, identity = prepare(jax, args)
        identity.update(device_kind=devices[0].device_kind, runtime_preflight_sha256=sha(args.runtime_preflight))
        journal.report.update(identity=identity, runtime_preflight=preflight, model_compiles=0)
        saved = read_pickle(args.old_snapshot)
        parameters, _ = validate_snapshot(jax, saved, manifest, "old")
        del saved
        device_parameters, device_samples = jax.device_put(parameters), jax.device_put(samples)
        arguments = (device_parameters, device_samples.inputs.continuous_state, device_samples)
        from jax.experimental.serialize_executable import serialize, deserialize_and_load
        abi = adjacent("shared_full_state_ad_execution")
        from scripts.hpc.run_shared_transformer_admission import memory_sizes
        executable_path = args.saved_executable or args.output.with_suffix(".executable.pkl")
        if args.saved_executable:
            sidecar = json.loads(executable_path.with_suffix(executable_path.suffix+".identity.json").read_text())
            if sidecar != {"identity": identity, "component": "daily"}:
                raise ValueError("corrected recovery exact identity differs")
            journal.emit("saved_corrected_executable_loading")
            with source.forbid_model_compile():
                serialized = read_pickle(executable_path)
                executable = deserialize_and_load(*serialized)
                abi.validate_full_state_abi(jax, serialized, executable, arguments, component="daily")
        else:
            if executable_path.exists() or executable_path.with_suffix(executable_path.suffix+".identity.json").exists():
                raise FileExistsError(executable_path)
            from research.daily_coarse_graining.daily_operator import shared_process_head
            from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_joint_daily_objective
            from jax_orchidee.stomate import carbon_kernels
            micro = adjacent("shared_daily_microbatch")
            dual = adjacent("shared_daily_dual_ad")
            with ExitStack() as patches:
                patches.enter_context(install_turnover_overlay(carbon_kernels, args.turnover_overlay, BASE_SHA))
                patches.enter_context(owner.candidate_process_head_patches(shared_process_head,
                    owner.load_staged_adapter(args.field_adapter), owner.load_backbone_assembler(args.backbone_adapter)))
                patches.enter_context(retain_row_diagnostics(micro))
                scalar = make_shared_joint_daily_objective(context["runtime"], context["head"], context["template"], context["state_scales"], context["science_scales"], include_state_diagnostics=True)
                backward = dual.make_daily_dual_value_and_grad(jax, dual.make_daily_dual_objective(jax, micro.make_microbatch_objective(jax, jnp, scalar)))
                journal.emit("lowering_corrected_graph")
                began = time.monotonic()
                with journal.deadline(args.lower_seconds, "lower"):
                    lowered = backward.lower(*arguments)
                journal.emit("backend_compiling_corrected_graph", lower_seconds=time.monotonic()-began)
                began = time.monotonic()
                journal.report["model_compiles"] = 1
                with journal.deadline(args.backend_seconds, "backend"):
                    executable = lowered.compile()
                # First postcompile action: preserve the sole expensive graph.
                serialized = serialize(executable)
                owner.atomic_pickle(executable_path, serialized)
                owner.atomic_json(executable_path.with_suffix(executable_path.suffix+".identity.json"), {"identity": identity, "component": "daily"})
                journal.emit("corrected_executable_serialized", executable=str(executable_path), compile_seconds=time.monotonic()-began)
                abi.validate_full_state_abi(jax, serialized, executable, arguments, component="daily")
        del serialized
        memory = memory_sizes(executable)
        journal.report["compiled_memory"] = memory
        if memory["estimated_live_bytes"] > 24*2**30:
            raise MemoryError("compiled memory exceeds24GiB")
        journal.emit("corrected_executable_memory_passed", compiled_memory=memory)
        for schedule in ("old", "new"):
            if schedule == "new":
                del device_parameters, arguments, parameters
                saved = read_pickle(args.new_snapshot)
                parameters, _ = validate_snapshot(jax, saved, manifest, "new")
                del saved
                device_parameters = jax.device_put(parameters)
                arguments = (device_parameters, device_samples.inputs.continuous_state, device_samples)
            first_path = None
            for repeat in range(1, args.repeats+1):
                journal.emit("corrected_call_entered", schedule=schedule, repeat=repeat)
                began = time.monotonic()
                with journal.deadline(args.call_seconds, "call_"+schedule+str(repeat)), source.forbid_model_compile():
                    host = jax.device_get(jax.block_until_ready(executable(*arguments)))
                path = args.output.with_suffix(f".{schedule}.repeat{repeat}.result.pkl")
                if path.exists(): raise FileExistsError(path)
                owner.atomic_pickle(path, {"identity": identity, "schedule": schedule, "repeat": repeat, "result": host})
                # Save first, validate second; no gradient information is dropped.
                validation = check_result(jax, host, parameters, samples)
                row = {"schedule": schedule, "repeat": repeat, "artifact": str(path), "seconds": time.monotonic()-began, "validation": validation}
                stats = devices[0].memory_stats()
                row["JAX_device_memory_stats"] = stats
                peak = stats.get("peak_bytes_in_use") if isinstance(stats, dict) else None
                row["observed_peak_bytes"] = peak
                row["observed_peak_memory_passed"] = isinstance(peak, (int, float)) and np.isfinite(peak) and 0 <= peak <= 24*2**30
                if first_path:
                    row["repeat_drift"] = compare_repeats(jax, read_pickle(first_path)["result"], host)
                else:
                    first_path = path
                journal.report["results"].append(row)
                del host
                journal.emit("corrected_call_preserved", **row)
                if peak is not None and peak > 24*2**30:
                    raise MemoryError("observed JAX peak memory exceeds24GiB")
        if args.original_reference:
            journal.emit("original_diagnostic_control_loading")
            original, evidence = abi.load_full_state_executable(jax=jax, path=args.original_reference,
                architecture="original", component="daily", admission=json.loads(Path(manifest["components"]["daily"]["report"]).read_text()),
                arguments=arguments, expected_reference=reference, compile_guard=source.forbid_model_compile)
            with journal.deadline(args.call_seconds, "original_control"), source.forbid_model_compile():
                control = jax.device_get(jax.block_until_ready(original(*arguments)))
            path = args.output.with_suffix(".original-new.control.pkl")
            owner.atomic_pickle(path, {"identity": identity, "schedule": "new", "result": control, "admission": evidence})
            journal.report["original_control"] = {"artifact": str(path), "scope": "immutable_uncorrected_new_snapshot_diagnostic_only"}
        results = journal.report["results"]
        journal.report.update(status="completed", finite_and_physical_passed=all(x["validation"]["finite"] and x["validation"]["physical_hard_counts_zero"] for x in results),
            meaningful_gradients_passed=all(x["validation"]["parameter_gradient_nonzero"] > 0 and x["validation"]["initial_state_gradient_nonzero"] > 0 for x in results),
            observed_peak_memory_passed=all(x["observed_peak_memory_passed"] for x in results),
            same_input_strict_parity=all(x.get("repeat_drift", {}).get("strict_parity", True) for x in results))
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
