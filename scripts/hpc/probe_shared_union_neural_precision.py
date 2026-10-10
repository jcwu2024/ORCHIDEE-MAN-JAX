"""Saved-FP32-union versus native TensorCore neural dots, full native AD.

Public objective and staged AD owners are probe_shared_neural_precision's
staged_daily_functions/native_staged_dispatcher. Physical process provenance,
losses, masks and model capacity remain with those existing source owners.
This candidate-only screen performs one full call and no warm timing replay.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("ORCHIDEE_REPO_ROOT", Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STAGE_NAMES = ("jit_neural_stage", "jit_transition_stage", "jit_loss_stage",
               "jit_loss_stage", "jit_transition_stage", "jit_neural_stage")
PHYSICAL_LOSS_ORDINALS = (1, 2, 3, 4)
PARAMETER_LEAVES = 300
PARAMETER_COUNT = 64_884_481
STATE_COUNT = 3854
FINITE_SCOPE = ["loss", "physical_outputs", "all_parameter_gradients", "all_continuous_state_gradients"]


class NumericalFailure(ValueError):
    """An unchanged physical or numerical result requirement failed."""


def execution_policy(cache_compatibility_only):
    return {"mode": "union" if cache_compatibility_only else "candidate",
            "required_cache_ordinals": tuple(range(6)) if cache_compatibility_only else PHYSICAL_LOSS_ORDINALS,
            "mixed": not cache_compatibility_only,
            "precision": "unchanged_FP32_neural_FP64_physical" if cache_compatibility_only else "F16_F16_F32_neural_dots"}


def strict_saved_result_comparison(jax, actual, reference):
    """Compare the complete validated value/auxiliary/gradient result tree."""
    left, left_tree = jax.tree_util.tree_flatten({name: actual[name] for name in ('loss', 'diagnostics', 'gradients')})
    right, right_tree = jax.tree_util.tree_flatten({name: reference[name] for name in ('loss', 'diagnostics', 'gradients')})
    if left_tree != right_tree:
        raise ValueError('cache replay/reference full result trees differ')
    failed = coordinates = paired_nan = 0
    for got, want in zip(left, right, strict=True):
        got, want = np.asarray(got), np.asarray(want)
        if got.shape != want.shape or got.dtype != want.dtype:
            raise ValueError('cache replay/reference full result shape/dtype differs')
        if np.issubdtype(got.dtype, np.inexact):
            close = np.isclose(got, want, rtol=1e-5, atol=1e-6, equal_nan=True)
            paired_nan += int(np.count_nonzero(np.isnan(got) & np.isnan(want)))
        else:
            close = got == want
        failed += int(np.count_nonzero(~close))
        coordinates += got.size
    return {'passed': failed == 0, 'leaves': len(left), 'coordinates': coordinates,
            'failed_coordinates': failed, 'paired_undefined_state_diagnostic_nan_coordinates': paired_nan,
            'rtol': 1e-5, 'atol': 1e-6, 'boolean_and_integer_comparison': 'exact',
            'NaN_pairing_scope': 'only_after_both_full_results_pass_defined_state_and_measured_finiteness_validation'}


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load diagnostic owner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_prior_report(report, identity):
    if (report.get("status") != "bounded_validity_passed" or report.get("mode") != "union"
            or report.get("precision") != "unchanged_FP32_neural_FP64_physical"
            or report.get("identity") != identity or report.get("neural_remat") is not True
            or report.get("loss_multiplier") != 1.0):
        raise ValueError("requires completed unchanged FP32 union native-AD report")
    units = report.get("compile_units", [])
    if len(units) != 6:
        raise ValueError("completed union report must contain all six native units")
    for ordinal, (unit, name) in enumerate(zip(units, STAGE_NAMES, strict=True)):
        if (unit.get("ordinal") != ordinal or unit.get("module") != name
                or unit.get("status") != "completed" or not unit.get("main_signature")):
            raise ValueError("completed native unit identity/signature is missing")
    return {"completed_units": 6, "mandatory_cache_hit_ordinals": list(PHYSICAL_LOSS_ORDINALS),
            "new_neural_unit_ordinals": [0, 5]}


def validate_full_result(jax, saved, identity, parameters, state, *, reference=False):
    if (saved.get("identity") != identity or saved.get("diagnostic_sample") != "prepared_micro4_row0"
            or saved.get("route") != "union"
            or saved.get("native_ad") != "all_parameters_and_continuous_state"):
        raise ValueError("result is not saved union row0 full native AD")
    if not reference and (saved.get("loss_multiplier") != 1.0 or saved.get("precision") != "F16_F16_F32_neural_dots"):
        raise ValueError("candidate policy/scaling differs")
    gradients = saved.get("gradients")
    if not isinstance(gradients, tuple) or len(gradients) != 2:
        raise ValueError("full parameter/state gradient pair is missing")
    p_leaves, p_tree = jax.tree_util.tree_flatten(parameters)
    g_leaves, g_tree = jax.tree_util.tree_flatten(gradients[0])
    if p_tree != g_tree or len(g_leaves) != PARAMETER_LEAVES or sum(np.size(x) for x in g_leaves) != PARAMETER_COUNT:
        raise ValueError("all 300 leaves/64884481 parameter gradients are required")
    for gradient, parameter in zip(g_leaves, p_leaves, strict=True):
        if np.shape(gradient) != np.shape(parameter) or np.asarray(gradient).dtype != np.asarray(parameter).dtype:
            raise ValueError("parameter gradient shape/dtype differs")
    if np.shape(state) != (STATE_COUNT,) or np.shape(gradients[1]) != (STATE_COUNT,) or np.asarray(gradients[1]).dtype != np.asarray(state).dtype:
        raise ValueError("all 3854 continuous-state gradients are required")
    diagnostics = saved["diagnostics"]
    if any(int(np.asarray(value)) != 0 for value in diagnostics["auxiliary"]["hard"].values()):
        raise NumericalFailure("physical hard gate failed")
    mask_schema = validate_mask_schema(jax, diagnostics)
    state_diagnostic_finiteness = validate_defined_state_diagnostics(jax, diagnostics)
    # Match the existing native-AD owner: undefined expected-state diagnostic
    # values are represented by their masks and may carry NaNs.
    measured = (saved["loss"], gradients, diagnostics["outputs"])
    if np.shape(saved["loss"]) != () or any(not np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(measured)):
        raise NumericalFailure("nonfinite loss, physical output or native gradient")
    if not any(np.any(np.asarray(x) != 0) for x in g_leaves) or not np.any(np.asarray(gradients[1]) != 0):
        raise NumericalFailure("meaningful parameter/state gradients are missing")
    return {"parameter_leaves": PARAMETER_LEAVES, "parameter_coordinates": PARAMETER_COUNT,
            "continuous_state_coordinates": STATE_COUNT, "all_measured_values_finite": True,
            "finite_scope": FINITE_SCOPE, "mask_schema": mask_schema,
            "state_diagnostic_finiteness": state_diagnostic_finiteness}


def validate_mask_schema(jax, diagnostics):
    masks = {"raw_masks": diagnostics["raw_masks"]}
    masks.update({key: diagnostics["auxiliary"]["state_diagnostics"][key]
                  for key in ("predicted_defined", "expected_defined")})
    result = {}
    for name, tree in masks.items():
        leaves = jax.tree_util.tree_leaves(tree)
        if not leaves or any(np.asarray(leaf).dtype != np.bool_ for leaf in leaves):
            raise ValueError(name + " mask tree must contain only boolean leaves")
        result[name] = {"leaves": len(leaves), "coordinates": sum(np.size(leaf) for leaf in leaves)}
    return result


def validate_defined_state_diagnostics(jax, diagnostics):
    state_diagnostics = diagnostics["auxiliary"]["state_diagnostics"]
    result = {}
    for name in ("predicted", "expected"):
        masks = {jax.tree_util.keystr(path): np.asarray(value) for path, value in
                 jax.tree_util.tree_flatten_with_path(state_diagnostics[name + "_defined"])[0]}
        values = jax.tree_util.tree_flatten_with_path(state_diagnostics[name])[0]
        if {jax.tree_util.keystr(path) for path, _ in values} != set(masks):
            raise ValueError("state diagnostic value/mask paths differ")
        nonfinite = nonfinite_defined = 0
        for path, value in values:
            value, mask = np.asarray(value), masks[jax.tree_util.keystr(path)]
            if value.shape != mask.shape:
                raise ValueError("state diagnostic value/mask shapes differ")
            invalid = ~np.isfinite(value)
            nonfinite += int(np.count_nonzero(invalid))
            nonfinite_defined += int(np.count_nonzero(invalid & mask))
        if nonfinite_defined:
            raise NumericalFailure("nonfinite declared-defined state diagnostic")
        result[name] = {"nonfinite": nonfinite, "nonfinite_defined": 0,
                        "nonfinite_undefined": nonfinite}
    return result


def clone_cache(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("candidate cache must be separate from preserved reference cache")
    if destination.exists():
        raise FileExistsError("candidate cache already exists; choose a new stem")
    if not source.is_dir() or not any(source.glob("jit_transition_stage-*-cache")) or not any(source.glob("jit_loss_stage-*-cache")):
        raise ValueError("preserved physical/loss executable cache is missing")
    started = time.perf_counter()
    shutil.copytree(source, destination)
    files = list(destination.rglob("*"))
    return {"source": str(source), "candidate": str(destination), "reference_cache_modified": False,
            "files": sum(path.is_file() for path in files),
            "bytes": sum(path.stat().st_size for path in files if path.is_file()),
            "copy_seconds": time.perf_counter() - started}


def validate_runtime_report(path, version):
    saved = json.loads(path.read_text(encoding="utf-8"))
    if (saved.get("status") != "passed" or saved.get("passed") is not True
            or saved.get("actual_backend") != "gpu" or saved.get("requested_backend") != "gpu"
            or saved.get("jax_version") != version):
        raise ValueError("deployed runtime preflight is missing, failed or incompatible")
    return {"artifact": str(path), "status": "passed", "jax_version": version}


@contextmanager
def compiler_totals(report):
    from jax._src import compiler
    original_lookup, original_backend = compiler.compile_or_get_cached, compiler.backend_compile
    active = []
    totals = {"lookup_calls": 0, "backend_entries": 0, "backend_completions": 0,
              "backend_seconds": 0.0, "by_module": {}}
    report["compiler_totals"] = totals

    def lookup(*args, **kwargs):
        module = args[1] if len(args) > 1 else kwargs["computation"]
        name = str(module.operation.attributes["sym_name"]).strip('"')
        totals["lookup_calls"] += 1
        row = totals["by_module"].setdefault(name, {"lookups": 0, "backend_entries": 0, "backend_completions": 0})
        row["lookups"] += 1
        active.append(name)
        try:
            return original_lookup(*args, **kwargs)
        finally:
            active.pop()

    def backend(*args, **kwargs):
        name = active[-1] if active else "outside_lookup"
        row = totals["by_module"].setdefault(name, {"lookups": 0, "backend_entries": 0, "backend_completions": 0})
        totals["backend_entries"] += 1
        row["backend_entries"] += 1
        started = time.perf_counter()
        result = original_backend(*args, **kwargs)
        totals["backend_completions"] += 1
        row["backend_completions"] += 1
        totals["backend_seconds"] += time.perf_counter() - started
        return result

    compiler.compile_or_get_cached, compiler.backend_compile = lookup, backend
    try:
        yield
    finally:
        compiler.compile_or_get_cached, compiler.backend_compile = original_lookup, original_backend


def run(args):
    started = time.perf_counter()
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(args.stage_cache)
    os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
    os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
    probe = load_script("union_tensorcore_native_owners", args.neural_probe_file)
    helper = probe.load_helper(args.real_daily_probe_file)
    cache_only = args.cache_compatibility_only
    policy = execution_policy(cache_only)
    report = {"status": "running", "scope": "saved_union_row0_full_parameter_and_state_AD_precision_screen",
              "contract": probe.CONTRACT, "mode": "candidate", "neural_remat": True,
              "loss_multiplier": 1.0, "training_updates": 0, "production_adoption": False,
              "scientific_acceptance": False, "row0_warm_timing_repeats": 0,
              "state_gradient_reference": "saved_FP32_union_same_execution_not_independent_field_parity",
              "inactive_gradient_gate_scope": {
                  "undefined_continuous_state_gradients": "exact_zero_checked",
                  "raw_and_state_masks": "exact_equality_checked",
                  "structural_inactive_parameter_and_output_coordinates": "not_separately_audited",
                  "arbitrary_reference_zero_coordinates": "descriptive_only_may_include_cancellation"},
              "finite_gate_scope": FINITE_SCOPE,
              "process_limit_seconds": args.process_seconds,
              "new_neural_unit_caps_seconds": [args.neural_forward_compile_seconds, args.final_neural_compile_seconds]}
    if cache_only:
        report.update(scope='saved_FP32_union_native_cache_compatibility_only',
                      cache_compatibility_only=True, cache_admission_eligible=False,
                      screening_passed=False, second_stage_eligible=False,
                      precision=policy['precision'], mode=policy['mode'], new_model_compilation_allowed=False,
                      new_neural_unit_caps_seconds=[], required_cache_hit_ordinals=list(range(6)))
    lock = threading.RLock()

    def emit(stage, **fields):
        with lock:
            elapsed = time.perf_counter() - started
            report.update(stage=stage, elapsed_seconds=elapsed, gpu_wall_exposure_card_hours=elapsed / 3600, **fields)
            helper.atomic_json(args.output, report)
            print(f"union_neural_precision {stage} seconds={elapsed:.3f}", flush=True)

    try:
        with probe.deadline(args.process_seconds, "whole_process", emit):
            emit("cache_copy_start")
            cache = clone_cache(args.reference_stage_cache, args.stage_cache)
            args.execution, args.mode, args.neural_remat = "staged", policy['mode'], True
            args.required_cache_ordinals = policy['required_cache_ordinals']
            args.forbid_named_model_backend = cache_only
            args.require_saved_ir_interface = cache_only
            probe.configure_stage_cache(args)
            import jax
            import jax.numpy as jnp
            with compiler_totals(report):
                from research.daily_coarse_graining.daily_operator import shared_process_head, shared_transformer
                from research.daily_coarse_graining.daily_operator.shared_recurrent_training import (
                    make_shared_joint_daily_objective,
                )
                from scripts.hpc.run_shared_learning import load_learning_checkpoint
                if len(jax.local_devices()) != 1 or jax.local_devices()[0].platform != "gpu":
                    raise ValueError("requires exactly one visible GPU")
                runtime = validate_runtime_report(args.runtime_preflight_report, jax.__version__)
                emit("imports_device_ready", cache=cache, runtime_preflight=runtime, jax_version=jax.__version__,
                     devices=[str(device) for device in jax.local_devices()], startup_seconds=time.perf_counter() - started)
                _, reference = helper.trusted_profile(args)
                identity = helper.context_identity(reference, helper.initial_checkpoint(reference), 742)
                if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != identity["source_git_head"]:
                    raise ValueError("immutable source checkout differs")
                with args.prepared_context.open("rb") as stream:
                    context = pickle.load(stream)
                if context["identity"] != identity:
                    raise ValueError("prepared context identity differs")
                checkpoint = load_learning_checkpoint(Path(identity["initial_checkpoint"]), expected_method=reference["method"])
                host_parameters = checkpoint["parameters"]
                host_sample = jax.tree_util.tree_map(lambda value: np.asarray(value)[0], context["daily4"])
                host_state = host_sample.inputs.continuous_state
                sample = jax.tree_util.tree_map(lambda value: jnp.asarray(value)[0], context["daily4"])
                with args.reference_result.open("rb") as stream:
                    saved_reference = pickle.load(stream)
                prior = json.loads(args.reuse_stage_report.read_text(encoding="utf-8"))
                prior_evidence = validate_prior_report(prior, identity)
                reference_evidence = validate_full_result(jax, saved_reference, identity, host_parameters, host_state, reference=True)
                if not probe.undefined_state_gradients_zero(jax, saved_reference, context):
                    raise ValueError("saved reference undefined state gradient differs")
                emit("saved_reference_ready", identity=identity, saved_reference_artifact=str(args.reference_result),
                     prior_cache_evidence=prior_evidence, reference_gradient_contract=reference_evidence)
                adapter = helper.load_staged_adapter(args.candidate_module_file)
                assembler = helper.load_backbone_assembler(args.backbone_module_file)
                package = "research.daily_coarse_graining.daily_operator."
                modules = [shared_transformer] + [sys.modules[package + name] for name in
                    ("shared_backbone_execution", "shared_backbone_projection_reuse", "shared_decoder_macro_packed")]
                captured = {}

                def captured_raw(*arguments, **keywords):
                    outputs, masks = adapter(*arguments, **keywords, dynamic_projection_execution="union")
                    captured["raw_masks"] = masks
                    return outputs, masks

                parameters = jax.device_put(host_parameters)
                state = sample.inputs.continuous_state
                stages = probe.staged_daily_functions(jax, context["head"], sample, context,
                    make_shared_joint_daily_objective, captured, type(context["head"]).learned_outputs, neural_remat=True)
                function = probe.native_staged_dispatcher(jax, stages)
                emit('cache_native_AD_start' if cache_only else "candidate_native_AD_start",
                     eligible_neural_modules=[module.__name__ for module in modules])
                began = time.perf_counter()
                with helper.candidate_process_head_patches(shared_process_head, captured_raw, assembler), probe.neural_policy(modules, mixed=policy['mixed']):
                    with probe.stage_compilation_evidence(args, helper, emit, report):
                        result = jax.block_until_ready(function(parameters, state, jnp.asarray(1.0, jnp.float32)))
                first_call = time.perf_counter() - began
                (loss, diagnostics), gradients = jax.device_get(result)
                saved = {"identity": identity, "diagnostic_sample": "prepared_micro4_row0", "route": "union",
                         "native_ad": "all_parameters_and_continuous_state", "loss": loss,
                         "diagnostics": diagnostics, "gradients": gradients, "loss_multiplier": 1.0,
                         "precision": policy['precision'], "neural_remat": True, "contract": probe.CONTRACT,
                         "cache_compatibility_only": cache_only}
                helper.atomic_pickle(args.result, saved)
                emit("candidate_result_saved", result_artifact=str(args.result),
                     cold_trace_compile_cache_persist_dispatch_execute_seconds=first_call)
                gradient_contract = validate_full_result(jax, saved, identity, host_parameters, host_state, reference=cache_only)
                for key in ("outputs", "raw_masks"):
                    if set(diagnostics[key]) != set(saved_reference["diagnostics"][key]):
                        raise ValueError("candidate/reference physical output or mask keys differ")
                units = report["compile_units"]
                if len(units) != 6 or any(units[i].get("required_cache_hit") is not True or units[i].get("cache_hit") is not True for i in policy['required_cache_ordinals']):
                    raise ValueError("all declared completed cache hits are required")
                stats = jax.local_devices()[0].memory_stats() or {}
                if stats.get("peak_bytes_in_use", 0) > args.memory_limit_gib * 2**30:
                    raise MemoryError("candidate active buffer peak exceeds declared memory cap")
                if cache_only:
                    if any(unit['backend_compile_calls'] != 0 or unit.get('native_cache_key_status') != 'observed' for unit in units):
                        raise ValueError('cache admission requires zero new model backend calls and actual native key evidence for all six units')
                    if not probe.undefined_state_gradients_zero(jax, saved, context):
                        raise NumericalFailure('cache replay undefined state gradient differs')
                    strict = strict_saved_result_comparison(jax, saved, saved_reference)
                    emit('completed', status='cache_compatibility_passed' if strict['passed'] else 'cache_compatibility_failed',
                         cache_admission_eligible=strict['passed'], screening_passed=False, second_stage_eligible=False,
                         strict_full_result_comparison=strict, cache_replay_gradient_contract=gradient_contract,
                         device_memory_stats=stats, new_model_backend_compile_calls=0)
                    return report
                comparison = probe.compare_results(jax, saved, saved_reference, context)
                emit("completed", status="screening_passed" if comparison["screening_passed"] else "screening_failed",
                     candidate_gradient_contract=gradient_contract, device_memory_stats=stats,
                     second_stage_eligible=comparison["screening_passed"], **comparison)
    except Exception as error:
        cache_failure = report.get('status') if report.get('status') in (
            'completed_unit_cache_miss', 'completed_unit_interface_mismatch', 'cache_reference_missing', 'model_backend_forbidden') else 'error'
        status = "resource_failed" if isinstance(error, MemoryError) else "numerical_failed" if isinstance(error, NumericalFailure) else cache_failure
        emit("exception", status=status,
             error_type=type(error).__name__, error=str(error))
        raise
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("source-profile", "prepared-context", "reference-result", "reference-stage-cache", "stage-cache",
                 "reuse-stage-report", "neural-probe-file", "real-daily-probe-file", "candidate-module-file",
                 "backbone-module-file", "runtime-preflight-report", "output", "result"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--process-seconds", type=int, default=1200)
    result.add_argument("--neural-forward-compile-seconds", type=int, default=300)
    result.add_argument("--final-neural-compile-seconds", type=int, default=600)
    result.add_argument("--memory-limit-gib", type=float, default=24)
    result.add_argument('--cache-compatibility-only', action='store_true',
                        help='FP32 union only; all six completed native caches must hit; never admits a precision screen')
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if (not 0 < arguments.process_seconds <= 1200 or not 0 < arguments.neural_forward_compile_seconds <= 300
            or not 0 < arguments.final_neural_compile_seconds <= 600 or not 0 < arguments.memory_limit_gib <= 24):
        raise ValueError("positive bounds within the prepared 1200/300/600s and 24GiB envelope are required")
    if arguments.cache_compatibility_only and arguments.process_seconds > 520:
        raise ValueError('cache compatibility process bound must be at most520 seconds')
    inputs = [value for name, value in vars(arguments).items() if isinstance(value, Path) and name not in ("output", "result", "stage_cache")]
    outputs = [arguments.output, arguments.result, arguments.stage_cache]
    if len({path.resolve() for path in outputs}) != 3 or any(path.resolve() in {out.resolve() for out in outputs} for path in inputs):
        raise ValueError("input/output collision")
    if any(path.exists() for path in outputs):
        raise FileExistsError("preserve diagnostic outputs; choose a new stem")
    run(arguments)
