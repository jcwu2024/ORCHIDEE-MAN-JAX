"""Original-four daily union versus admitted combined field binary, no training.

Objective owners: shared_recurrent_training::make_shared_joint_daily_objective
and shared_daily_microbatch::make_microbatch_objective. Only the opt-in expert
union adapter and source-reuse backbone are bound during native AD lowering.
The returned contract is the production scalar/auxiliary/all-parameter gradient.
State differentiability and all physical/mask computations remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("ORCHIDEE_REPO_ROOT", Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PARAMETER_COUNT = 64_884_481
PARAMETER_LEAVES = 300
RTOL, ATOL = 1e-5, 1e-6
VARIANT = "expert128+backbone_source_reuse"


class NumericalFailure(ValueError):
    """An unchanged numerical or physical admission gate failed."""


def load_owner(path: Path):
    spec = importlib.util.spec_from_file_location("union_micro4_daily_owner", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load daily helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def deadline(seconds: float, label: str, emit):
    # A process exit also bounds native compilation that cannot be interrupted.
    def expired():
        emit("timeout", status="timeout", timeout_scope=label, limit_seconds=seconds)
        os._exit(124)

    timer = threading.Timer(seconds, expired)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


@contextmanager
def forbid_field_compile():
    from jax._src import compiler

    original = compiler.compile_or_get_cached

    def forbidden(*args, **kwargs):
        raise RuntimeError("saved field replay attempted a new graph executable lookup")

    compiler.compile_or_get_cached = forbidden
    try:
        yield
    finally:
        compiler.compile_or_get_cached = original


def validate_original_rows(jax, daily4, original_daily):
    leaves, tree = jax.tree_util.tree_flatten(daily4)
    original, original_tree = jax.tree_util.tree_flatten(original_daily)
    if tree != original_tree or not leaves:
        raise ValueError("prepared and original sample pytrees differ")
    hashes = [hashlib.sha256() for _ in range(4)]
    for value, source in zip(leaves, original, strict=True):
        value, source = np.asarray(value), np.asarray(source)
        if not value.shape or value.shape[0] != 4 or not source.shape or source.shape[0] != 32:
            raise ValueError("requires prepared four rows and original 32-row bundle")
        if value.dtype != source.dtype or value.shape[1:] != source.shape[1:]:
            raise ValueError("original sample shape/dtype differs")
        if not np.array_equal(value, source[:4], equal_nan=True):
            raise ValueError("prepared rows differ from the original first four bundle rows")
        for row, digest in enumerate(hashes):
            digest.update(str((value.shape[1:], value.dtype.str)).encode("ascii"))
            digest.update(np.ascontiguousarray(value[row]).tobytes())
    fingerprints = [digest.hexdigest() for digest in hashes]
    if len(set(fingerprints)) != 4:
        raise ValueError("requires four distinct original rows, never repeated row0")
    return {"row_indices": [0, 1, 2, 3], "row_sha256": fingerprints, "sample_leaves": len(leaves)}


def validate_binary_inputs(jax, serialized, executable, parameters, daily4):
    if not isinstance(serialized, tuple) or len(serialized) != 3:
        raise ValueError("saved executable is not a JAX serialized triple")
    supplied = ((parameters, daily4), {})
    if serialized[1] != jax.tree_util.tree_structure(supplied):
        raise ValueError("saved binary dynamic input treedef differs")
    infos, info_tree = jax.tree_util.tree_flatten(executable.args_info)
    values, supplied_tree = jax.tree_util.tree_flatten(supplied)
    if info_tree != supplied_tree or len(infos) != len(values):
        raise ValueError("loaded binary dynamic argument tree differs")
    for info, value in zip(infos, values, strict=True):
        aval = info._aval
        if tuple(aval.shape) != np.shape(value) or np.dtype(aval.dtype) != np.dtype(value.dtype):
            raise ValueError("saved binary dynamic input shape/dtype differs")
    return {"treedef_matches": True, "shapes_dtypes_match": True, "dynamic_leaves": len(values)}


def validate_admission(report, expected_identity):
    if (
        report.get("status") != "passed"
        or report.get("identity") != expected_identity
        or report.get("variant") != VARIANT
        or report.get("rtol") != RTOL
        or report.get("atol") != ATOL
        or report.get("all_parameter_gradient_coordinates") != PARAMETER_COUNT
    ):
        raise ValueError("baseline admission method/variant/strict gate differs")
    for name in ("loss", "auxiliary", "all_parameter_gradients"):
        if report.get("comparisons", {}).get(name, {}).get("passed") is not True:
            raise ValueError("baseline lacks completed strict numerical admission")
    return Path(report["candidate_executable"]), Path(report["candidate_artifact"])


def validate_runtime_report(path, jax_version):
    saved = json.loads(path.read_text(encoding="utf-8"))
    if (
        saved.get("status") != "passed"
        or saved.get("passed") is not True
        or saved.get("actual_backend") != "gpu"
        or saved.get("requested_backend") != "gpu"
        or saved.get("jax_version") != jax_version
    ):
        raise ValueError("deployed runtime preflight is missing, failed or incompatible")
    return {"artifact": str(path), "status": "passed", "jax_version": jax_version}


def load_reference(path, expected_identity, *, variant, legacy_canonical_path=None):
    with path.open("rb") as stream:
        saved = pickle.load(stream)
    legacy_canonical = (
        variant == "canonical_reference"
        and "variant" not in saved
        and legacy_canonical_path is not None
        and path.resolve() == legacy_canonical_path.resolve()
        and expected_identity.get("artifact_id") == 742
        and expected_identity.get("daily_rows") == 4
    )
    if (
        saved.get("identity") != expected_identity
        or (saved.get("variant") != variant and not legacy_canonical)
        or "diagnostic_sample" in saved.get("identity", {})
    ):
        raise ValueError("reference is not the admitted original-four result")
    result = saved["result"]
    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], tuple) or len(result[0]) != 2:
        raise ValueError("reference result lacks scalar/auxiliary/parameter-gradient contract")
    provenance = {
        "artifact": str(path.resolve()),
        "expected_variant": variant,
        "variant_tag": saved.get("variant"),
        "provenance": "legacy_untagged_canonical_reference_from_admission"
        if legacy_canonical
        else "explicit_variant_tag",
        "admitted_reference_artifact": str(legacy_canonical_path.resolve()) if legacy_canonical else None,
    }
    return result, provenance


def validate_result(jax, result, parameters, *, count=PARAMETER_COUNT, leaves=PARAMETER_LEAVES):
    if not isinstance(result, tuple) or len(result) != 2 or len(result[0]) != 2:
        raise ValueError("result does not match scalar/auxiliary/parameter-gradient contract")
    (loss, auxiliary), gradients = result
    if np.shape(loss) != () or not np.isfinite(np.asarray(loss)):
        raise NumericalFailure("daily scalar loss is invalid")
    hard = auxiliary.get("hard")
    if not isinstance(hard, dict) or not hard:
        raise ValueError("canonical physical hard counts missing")
    for name, value in hard.items():
        array = np.asarray(value)
        if array.shape or not np.isfinite(array) or array < 0 or array != np.floor(array):
            raise NumericalFailure(f"invalid physical hard count: {name}")
    if any(int(np.asarray(value)) for value in hard.values()):
        raise NumericalFailure(f"physical hard gate failed: {hard}")
    for value in jax.tree_util.tree_leaves(auxiliary):
        if not np.all(np.isfinite(np.asarray(value))):
            raise NumericalFailure("canonical auxiliary contains nonfinite values")
    gradient_leaves, gradient_tree = jax.tree_util.tree_flatten(gradients)
    parameter_leaves, parameter_tree = jax.tree_util.tree_flatten(parameters)
    if gradient_tree != parameter_tree or len(gradient_leaves) != leaves:
        raise ValueError("full parameter-gradient tree/leaves differ")
    coordinates = nonzero = 0
    norm_squared = 0.0
    for gradient, parameter in zip(gradient_leaves, parameter_leaves, strict=True):
        gradient, parameter = np.asarray(gradient), np.asarray(parameter)
        if gradient.shape != parameter.shape or gradient.dtype != parameter.dtype:
            raise ValueError("parameter gradient shape/dtype differs")
        if not np.all(np.isfinite(gradient)):
            raise NumericalFailure("parameter gradient contains nonfinite values")
        coordinates += gradient.size
        nonzero += int(np.count_nonzero(gradient))
        flat = gradient.astype(np.float64)
        norm_squared += float(np.sum(flat * flat))
    if coordinates != count or not nonzero or not np.isfinite(norm_squared) or norm_squared <= 0:
        raise NumericalFailure("full meaningful parameter-gradient admission failed")
    return {
        "gradient_leaves": leaves,
        "gradient_coordinates": coordinates,
        "gradient_nonzero": nonzero,
        "gradient_norm": float(np.sqrt(norm_squared)),
        "hard_counts": {name: int(np.asarray(value)) for name, value in hard.items()},
    }


def comparisons(helper, jax, actual, reference):
    return {
        "loss": helper.compare_trees(jax, actual[0][0], reference[0][0]),
        "auxiliary": helper.compare_trees(jax, actual[0][1], reference[0][1]),
        "all_parameter_gradients": helper.compare_trees(jax, actual[1], reference[1]),
    }


def require_parity(values):
    if not all(value["passed"] for value in values.values()):
        raise NumericalFailure("strict scalar/auxiliary/all-parameter parity gate failed")


def validate_precision_screen(path, neural_probe, identity):
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("status") != "screening_passed" or report.get("screening_passed") is not True
            or report.get("second_stage_eligible") is not True or report.get("identity") != identity
            or report.get("contract") != neural_probe.CONTRACT or report.get("loss_multiplier") != 1.0
            or any(report.get("candidate_gradient_contract", {}).get(key) != value for key, value in
                   {"parameter_leaves": PARAMETER_LEAVES, "parameter_coordinates": PARAMETER_COUNT,
                    "continuous_state_coordinates": 3854, "all_measured_values_finite": True}.items())
            or report.get("candidate_gradient_contract", {}).get("finite_scope") !=
                ["loss", "physical_outputs", "all_parameter_gradients", "all_continuous_state_gradients"]
            or report.get("scope") != "saved_union_row0_full_parameter_and_state_AD_precision_screen"):
        raise ValueError("mixed micro4 requires completed unchanged full parameter/state engineering screen")
    return {"artifact": str(path), "full_parameter_and_state_screening_passed": True}


def precision_comparisons(neural_probe, jax, actual, reference):
    groups = neural_probe.parameter_groups(actual[1])
    reference_groups = neural_probe.parameter_groups(reference[1])
    metrics = {name: neural_probe.vector_metrics(jax, value, reference_groups[name])
               for name, value in groups.items()}
    expected = abs(float(np.asarray(reference[0][0])))
    delta = abs(float(np.asarray(actual[0][0]) - np.asarray(reference[0][0])))
    loss_passed = delta <= neural_probe.CONTRACT["near_zero_absolute_max_limit"] if expected <= neural_probe.CONTRACT["near_zero_reference_norm"] else delta / expected <= neural_probe.CONTRACT["loss_relative_limit"]
    return {"contract": neural_probe.CONTRACT, "parameters": metrics, "loss_absolute_difference": delta,
            "loss_relative_difference": None if expected <= neural_probe.CONTRACT["near_zero_reference_norm"] else delta / expected,
            "screening_passed": bool(loss_passed and all(row["screening_passed"] for row in metrics.values()))}


def synchronized_call(jax, executable, parameters, daily4):
    began = time.perf_counter()
    result = executable(parameters, daily4)
    jax.block_until_ready(result)
    return result, time.perf_counter() - began


def positive_seconds(value):
    seconds = float(value)
    if not np.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("time budgets must be finite and positive")
    return seconds


def matched_warm(jax, field, union, parameters, daily4, emit, call_seconds):
    times = {"field": [], "union": []}
    for name, executable in (("field", field), ("union", union)):
        with deadline(call_seconds, name + "_explicit_warmup", emit):
            result, seconds = synchronized_call(jax, executable, parameters, daily4)
            del result
        emit(name + "_explicit_warmup", **{name + "_warmup_seconds": seconds})
    for repeat in range(3):
        order = ("field", "union") if repeat % 2 == 0 else ("union", "field")
        for name in order:
            with deadline(call_seconds, f"{name}_warm_{repeat}", emit):
                result, seconds = synchronized_call(jax, field if name == "field" else union, parameters, daily4)
                del result
            times[name].append(seconds)
            emit("matched_warm_repeat", repeat=repeat, order=list(order), matched_warm_seconds=times)
    return times


def run(args):
    started = time.perf_counter()
    helper = load_owner(args.real_daily_probe_file)
    outputs = (
        args.output,
        args.output.with_suffix(".candidate.pkl"),
        args.output.with_suffix(".candidate.executable.pkl"),
        args.output.with_suffix(".field-replay.pkl"),
    )
    if any(path.exists() for path in outputs):
        raise ValueError("output already exists; preserve prior artifacts and choose a new output stem")
    helper.check_artifact_output_collisions(
        (
            args.source_profile,
            args.prepared_context,
            args.input_bundle,
            args.baseline_admission_report,
            args.reference_result,
            args.canonical_reference_result,
            args.real_daily_probe_file,
            args.candidate_module_file,
            args.backbone_module_file,
            args.runtime_preflight_report,
            args.neural_probe_file,
            args.required_precision_screen,
        ),
        outputs,
    )
    report = {
        "status": "running",
        "scope": "original_four_daily_micro4_parameter_AD_only",
        "production_adoption": False,
        "scientific_acceptance": False,
        "training_updates": 0,
        "optimizer": False,
        "T7_T30": False,
        "state_gradient_parity": "not_established",
        "separate_row0_full_state_admission_retained": True,
        "prior_component_failed_coordinates": 27,
        "rtol": RTOL,
        "atol": ATOL,
        "precision": "unchanged_FP32_neural_FP64_physical",
        "gpu_count": 1,
        "budgets_seconds": {
            "lower": args.lower_seconds,
            "backend": args.backend_seconds,
            "process": args.process_seconds,
            "first_execution": args.first_execution_seconds,
        },
    }
    emit_lock = threading.RLock()

    def emit(stage, **fields):
        with emit_lock:
            elapsed = time.perf_counter() - started
            report.update(stage=stage, elapsed_seconds=elapsed, gpu_wall_exposure_card_hours=elapsed / 3600, **fields)
            helper.atomic_json(args.output, report)
            print(f"union_micro4 {stage} seconds={elapsed:.3f}", flush=True)

    try:
        with deadline(args.process_seconds, "whole_process", emit):
            import jax
            import jax.numpy as jnp
            from jax.experimental.serialize_executable import deserialize_and_load, serialize

            from research.daily_coarse_graining.daily_operator import shared_process_head
            from research.daily_coarse_graining.daily_operator.shared_recurrent_training import (
                make_shared_joint_daily_objective,
            )
            from scripts.hpc.probe_shared_compile_cache import _require_supported_jax
            from scripts.hpc.run_shared_learning import load_learning_checkpoint
            from scripts.hpc.run_shared_transformer_admission import memory_sizes
            from scripts.hpc.shared_daily_microbatch import make_microbatch_objective

            _require_supported_jax(jax)
            runtime_preflight = validate_runtime_report(args.runtime_preflight_report, jax.__version__)
            if len(jax.local_devices()) != 1 or jax.local_devices()[0].platform != "gpu":
                raise ValueError("requires exactly one visible GPU for saved binary compatibility")
            emit(
                "imports_device_ready",
                cold_startup_seconds=time.perf_counter() - started,
                jax_version=jax.__version__,
                devices=[str(device) for device in jax.local_devices()],
                runtime_preflight=runtime_preflight,
            )
            began = time.perf_counter()
            profile, reference = helper.trusted_profile(args)
            if (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
                != reference["source_git_head"]
            ):
                raise ValueError("immutable source checkout differs")
            identity = helper.context_identity(reference, helper.initial_checkpoint(reference), 742)
            neural_probe = None
            if args.neural_probe_file is not None:
                if args.required_precision_screen is None:
                    raise ValueError("TensorCore micro4 requires full-state screen report")
                neural_probe = load_owner(args.neural_probe_file)
                gate = validate_precision_screen(args.required_precision_screen, neural_probe, identity)
                report.update(precision="F16_F16_F32_neural_dots", prerequisite_full_state_screen=gate)
                report["inactive_gradient_gate_scope"] = {
                    "undefined_state_and_raw_state_masks": "passing_first_stage_bounded_evidence",
                    "structural_inactive_parameter_and_output_coordinates": "not_separately_audited",
                    "arbitrary_reference_zero_coordinates": "descriptive_only_may_include_cancellation"}
            elif args.required_precision_screen is not None:
                raise ValueError("precision screen requires explicit neural policy owner")
            if int(reference["method"]["training_plan"]["training_artifact_ids"][0]) != 742:
                raise ValueError("expected original artifact742")
            canonical_binary = Path(profile["executables"]["daily"]["executable"]).resolve()
            legacy_identity = identity | {"jax_version": jax.__version__, "executable": str(canonical_binary)}
            with args.prepared_context.open("rb") as stream:
                context = pickle.load(stream)
            if context["identity"] != identity or Path(context["daily_executable"]).resolve() != canonical_binary:
                raise ValueError("prepared method/checkpoint/canonical binary identity differs")
            with args.input_bundle.open("rb") as stream:
                bundle = pickle.load(stream)
            if (
                bundle.get("schema_version") != "shared_serialized_dual_inputs_v1"
                or bundle.get("method") != reference["method"]
                or bundle.get("source_git_head") != identity["source_git_head"]
                or tuple(bundle.get("artifact_ids", ()))
                != tuple(reference["method"]["training_plan"]["training_artifact_ids"][:2])
                or len(bundle.get("payloads", ())) != 2
                or Path(bundle.get("initial_checkpoint", "")).resolve() != Path(identity["initial_checkpoint"])
            ):
                raise ValueError("original compact bundle method/checkpoint/artifacts differ")
            rows = validate_original_rows(jax, context["daily4"], bundle["payloads"][0]["daily"])
            del bundle
            admission = json.loads(args.baseline_admission_report.read_text(encoding="utf-8"))
            baseline_path, saved_path = validate_admission(admission, legacy_identity)
            reference_path = args.reference_result or saved_path
            helper.check_artifact_output_collisions(
                (baseline_path, reference_path, canonical_binary, Path(identity["initial_checkpoint"])), outputs
            )
            saved_reference, baseline_reference_provenance = load_reference(
                reference_path, legacy_identity, variant=VARIANT
            )
            canonical_reference = canonical_reference_provenance = None
            if args.canonical_reference_result:
                canonical_reference, canonical_reference_provenance = load_reference(
                    args.canonical_reference_result,
                    legacy_identity,
                    variant="canonical_reference",
                    legacy_canonical_path=Path(admission["reference_artifact"]),
                )
            checkpoint = load_learning_checkpoint(
                Path(identity["initial_checkpoint"]), expected_method=reference["method"]
            )
            host_parameters = checkpoint["parameters"]
            parameter_leaves = jax.tree_util.tree_leaves(host_parameters)
            if (
                len(parameter_leaves) != PARAMETER_LEAVES
                or sum(np.size(value) for value in parameter_leaves) != PARAMETER_COUNT
            ):
                raise ValueError("update0 full parameter count/leaves differ")
            if any(np.asarray(value).dtype != np.float32 for value in parameter_leaves):
                raise ValueError("update0 neural parameter precision differs")
            parameters, daily4 = jax.device_put((host_parameters, context["daily4"]))
            jax.block_until_ready((parameters, daily4))
            result_identity = legacy_identity | {
                "comparison_schema": "shared_union_original_micro4_v1",
                "original_rows": rows,
                "baseline_binary": str(baseline_path),
            }
            if neural_probe is not None:
                result_identity["candidate_precision"] = "F16_F16_F32_neural_dots"
            emit(
                "inputs_reference_ready",
                identity=result_identity,
                reference_loading_seconds=time.perf_counter() - began,
                baseline_variant=VARIANT,
                baseline_binary=str(baseline_path),
                canonical_provenance_binary=str(canonical_binary),
                reference_artifact=str(reference_path),
                baseline_reference_provenance=baseline_reference_provenance,
                canonical_reference_provenance=canonical_reference_provenance,
            )
            limit = args.memory_limit_gib * 2**30
            with forbid_field_compile():
                began = time.perf_counter()
                with baseline_path.open("rb") as stream:
                    serialized = pickle.load(stream)
                binary_loading_seconds = time.perf_counter() - began
                began = time.perf_counter()
                field = deserialize_and_load(*serialized)
                field_deserialize_seconds = time.perf_counter() - began
                binary_inputs = validate_binary_inputs(jax, serialized, field, parameters, daily4)
                del serialized
                field_memory = memory_sizes(field)
                emit(
                    "field_deserialized",
                    field_deserialize_seconds=field_deserialize_seconds,
                    field_binary_loading_seconds=binary_loading_seconds,
                    field_memory=field_memory,
                    binary_inputs=binary_inputs,
                    field_graph_compiles=0,
                )
                if field_memory["estimated_live_bytes"] > limit:
                    raise MemoryError("field executable exceeds 24GiB admission")
                with deadline(args.first_execution_seconds, "field_first_execution", emit):
                    value, seconds = synchronized_call(jax, field, parameters, daily4)
                    field_result = jax.device_get(value)
                    del value
                helper.atomic_pickle(
                    outputs[3], {"identity": result_identity, "variant": VARIANT, "result": field_result}
                )
                emit("field_result_saved", field_first_execution_seconds=seconds, field_replay_artifact=str(outputs[3]))
                field_admission = validate_result(jax, field_result, host_parameters)
                field_stats = jax.local_devices()[0].memory_stats() or {}
                if field_stats.get("peak_bytes_in_use", 0) > limit:
                    raise MemoryError("observed field device peak exceeds 24GiB admission")
                saved_parity = comparisons(helper, jax, field_result, saved_reference)
                emit(
                    "field_saved_reference_compared",
                    field_saved_reference_comparisons=saved_parity,
                    field_admission=field_admission,
                    field_device_memory_stats=field_stats,
                    field_observed_peak_available="peak_bytes_in_use" in field_stats,
                )
                require_parity(saved_parity)
            del saved_reference
            adapter = partial(
                helper.load_staged_adapter(args.candidate_module_file), dynamic_projection_execution="union"
            )
            backbone = helper.load_backbone_assembler(args.backbone_module_file)
            daily_objective = make_shared_joint_daily_objective(
                context["runtime"],
                context["head"],
                context["template"],
                context["state_scales"],
                context["science_scales"],
            )
            objective = make_microbatch_objective(jax, jnp, daily_objective)
            backward = jax.jit(jax.value_and_grad(objective, has_aux=True))
            policy = nullcontext()
            if neural_probe is not None:
                package = "research.daily_coarse_graining.daily_operator."
                modules = [sys.modules[package + name] for name in
                           ("shared_transformer", "shared_backbone_execution", "shared_backbone_projection_reuse", "shared_decoder_macro_packed")]
                policy = neural_probe.neural_policy(modules, mixed=True)
            with helper.candidate_process_head_patches(shared_process_head, adapter, backbone), policy:
                emit("union_before_lower")
                with deadline(args.lower_seconds, "union_lower", emit):
                    began = time.perf_counter()
                    lowered = backward.lower(parameters, daily4)
                    lower_seconds = time.perf_counter() - began
                emit("union_before_backend_compile", union_lower_seconds=lower_seconds)
                with deadline(args.backend_seconds, "union_backend_compile", emit):
                    began = time.perf_counter()
                    union = lowered.compile()
                    compile_seconds = time.perf_counter() - began
            helper.atomic_pickle(outputs[2], serialize(union))
            union_memory = memory_sizes(union)
            emit(
                "union_executable_saved",
                union_backend_seconds=compile_seconds,
                union_memory=union_memory,
                candidate_executable=str(outputs[2]),
            )
            if union_memory["estimated_live_bytes"] > limit:
                raise MemoryError("union executable exceeds 24GiB admission")
            with deadline(args.first_execution_seconds, "union_first_execution", emit):
                value, seconds = synchronized_call(jax, union, parameters, daily4)
                union_result = jax.device_get(value)
                del value
            helper.atomic_pickle(
                outputs[1],
                {"identity": result_identity, "variant": "union+backbone_source_reuse",
                 "precision": report["precision"], "result": union_result},
            )
            emit("union_result_saved", union_first_execution_seconds=seconds, candidate_artifact=str(outputs[1]))
            stats = jax.local_devices()[0].memory_stats() or {}
            if stats.get("peak_bytes_in_use", 0) > limit:
                raise MemoryError("observed process device peak exceeds 24GiB admission")
            admission_values = validate_result(jax, union_result, host_parameters)
            parity = comparisons(helper, jax, union_result, field_result)
            canonical_parity = (
                comparisons(helper, jax, union_result, canonical_reference) if canonical_reference is not None else None
            )
            emit(
                "union_strict_compared",
                union_admission=admission_values,
                comparisons=parity,
                canonical_comparisons=canonical_parity,
                device_memory_stats=stats,
                union_observed_peak_available="peak_bytes_in_use" in stats,
            )
            if neural_probe is None:
                require_parity(parity)
                if canonical_parity is not None:
                    require_parity(canonical_parity)
            else:
                engineering = precision_comparisons(neural_probe, jax, union_result, field_result)
                canonical_engineering = precision_comparisons(neural_probe, jax, union_result, canonical_reference) if canonical_reference is not None else None
                emit("union_engineering_compared", engineering_comparisons=engineering,
                     canonical_engineering_comparisons=canonical_engineering)
                if not engineering["screening_passed"] or (canonical_engineering is not None and not canonical_engineering["screening_passed"]):
                    raise NumericalFailure("unchanged precision engineering gate failed; no warm timing")
            del union_result, field_result, canonical_reference
            emit("gates_passed", time_to_valid_measurement_seconds=time.perf_counter() - started)
            with forbid_field_compile():
                warm = matched_warm(jax, field, union, parameters, daily4, emit, args.first_execution_seconds)
            emit(
                "completed",
                status="passed",
                matched_warm_seconds=warm,
                field_over_union_median_ratio=float(np.median(warm["field"]) / np.median(warm["union"])),
                timing_scope="synchronized_original_four_daily_parameter_AD_not_complete_update_throughput",
            )
    except Exception as error:
        status = (
            "resource_failed"
            if isinstance(error, MemoryError)
            else "numerical_failed"
            if isinstance(error, NumericalFailure)
            else "error"
        )
        emit("exception", status=status, error_type=type(error).__name__, error=str(error))
        raise
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-profile",
        "prepared-context",
        "input-bundle",
        "baseline-admission-report",
        "real-daily-probe-file",
        "candidate-module-file",
        "backbone-module-file",
        "runtime-preflight-report",
        "output",
    ):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--reference-result", type=Path)
    result.add_argument("--canonical-reference-result", type=Path)
    result.add_argument("--neural-probe-file", type=Path)
    result.add_argument("--required-precision-screen", type=Path)
    for name, default in (("lower", 180), ("backend", 900), ("process", 1500), ("first-execution", 100)):
        result.add_argument("--" + name + "-seconds", type=positive_seconds, default=default)
    result.add_argument("--memory-limit-gib", type=float, default=24)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if not 0 < arguments.memory_limit_gib <= 24:
        raise SystemExit("memory limit must be positive and at most 24GiB")
    run(arguments)
