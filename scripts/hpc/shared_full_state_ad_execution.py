"""Admission and canonical-call adapters for serialized native full-state AD.

Execution contract follows probe_shared_latent_reader.py: native argnums=(0,1),
daily explicit micro4 states, and recurrent five-element cases. This adapter
changes only the Python return view; every call executes and checks state AD.
It does not establish scientific acceptance or compile a model.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np

CANDIDATE_ARCHITECTURE = "shared_latent32x384_reader64x2"
ORIGINAL_ARCHITECTURE = "original"
SCIENTIFIC_ADMISSION = "scientific"
CORRECTED_PERFORMANCE_DIAGNOSTIC = "corrected_performance_diagnostic"


def json_identity(value):
    """Match the JSON identity representation used by executable sidecars."""
    return json.loads(json.dumps(value, default=str))


def state_shape(component):
    if component == "daily":
        return (4, 3854)
    if component in ("T7", "T30"):
        return (3854,)
    raise ValueError("component must be daily, T7 or T30")


def _hard_gate(report):
    hard = report.get("hard_counts")
    if not isinstance(hard, dict) or not hard or any(value != 0 for value in hard.values()):
        raise ValueError("admission hard-count gates missing or failed")


def validate_candidate_admission(report, component, expected_identity):
    """Require this candidate's own component technical admission."""
    shape = state_shape(component)
    valid = report.get("validation", {})
    identity = report.get("identity", {})
    config = identity.get("candidate_config", {})
    if (report.get("status") != "passed" or report.get("component") != component
            or report.get("architecture") != CANDIDATE_ARCHITECTURE
            or report.get("native_AD_argnums") != [0, 1]
            or report.get("initialization") != "fresh_public_whole_tree"
            or report.get("training_updates") != 0
            or report.get("precision") != "FP32_neural_FP64_physics"
            or config.get("latent_reader_execution") != "field"
            or config.get("decoder_architecture") != "latent_reader"
            or not identity.get("candidate_source_sha256")
            or not identity.get("source_reuse_assembler_sha256")
            or not identity.get("source_reuse_projection_sha256")
            or not identity.get("source_head")
            or not identity.get("input_source_method")
            or json_identity(report.get("identity")) != json_identity(expected_identity)
            or report.get("parameter_count") != 66123713
            or report.get("parameter_leaves") != 355
            or valid.get("gradient_coordinates") != 66123713
            or valid.get("gradient_leaves") != 355
            or not valid.get("gradient_nonzero", 0) > 0
            or not np.isfinite(valid.get("gradient_norm", np.nan))
            or tuple(valid.get("initial_state_shape", ())) != shape
            or valid.get("initial_state_finite") is not True
            or not valid.get("initial_state_nonzero", 0) > 0):
        raise ValueError("candidate full-state component admission differs")
    _hard_gate(valid)
    return report


def validate_original_admission(report, component, expected_reference):
    """Admit only original full-state evidence, including historical V4 reference.

    The V4 daily filename contains the field_original reference; candidate V4
    results must never enter this original architecture route.
    """
    shape = state_shape(component)
    identity = report.get("identity", {})
    valid = report.get("validation", report)
    scope = ("original_four_daily_micro4_public_loss_full_parameter_and_initial_state_AD"
             if component == "daily" else "public_recurrent_loss_all_parameters_and_initial_state_AD")
    full = report.get("parameter_and_initial_state_gradients") is True or report.get("scope") == scope
    if (report.get("status") != "passed" or not full
            or (component == "daily" and report.get("variant") != "field_original")
            or (component != "daily" and report.get("horizon") != int(component[1:]))
            or (report.get("component") is not None and report["component"] != component)
            or identity.get("source_head", identity.get("source_git_head")) != expected_reference["source_git_head"]
            or json_identity(identity.get("method")) != json_identity(expected_reference["method"])
            or valid.get("initial_state_gradient_finite") is not True
            or valid.get("initial_state_gradient_coordinates") != int(np.prod(shape))
            or not valid.get("initial_state_gradient_nonzero", 0) > 0):
        raise ValueError("original full-state reference admission differs")
    _hard_gate(valid)
    return report


def validate_corrected_performance_diagnostic(report, component, architecture, *,
                                              expected_identity=None, expected_reference=None,
                                              expected_turnover_sha256=None,
                                              expected_executable=None,
                                              expected_prepared_context_identity=None):
    """Validate saved matched diagnostic evidence without scientific admission.

    The source report remains ``numerical_failed`` when strict repeat state
    comparisons failed. This route admits only its finite full-AD executable
    inputs for a separately bounded performance diagnostic.
    """
    if architecture not in (ORIGINAL_ARCHITECTURE, CANDIDATE_ARCHITECTURE):
        raise ValueError("explicit supported architecture required")
    if (report.get("status") not in ("passed", "numerical_failed")
            or report.get("component") != component
            or report.get("native_AD_argnums") != [0, 1]
            or report.get("training_updates") != 0
            or report.get("precision") != "FP32_neural_FP64_physics"
            or report.get("scientific_acceptance") is not False
            or report.get("model_compiles", {}).get(
                "candidate" if architecture == CANDIDATE_ARCHITECTURE
                and "candidate" in report.get("identities", {}) else architecture) != 0):
        raise ValueError("corrected performance diagnostic report status/scope differs")
    report_architecture = ("candidate" if architecture == CANDIDATE_ARCHITECTURE
                           and "candidate" in report.get("identities", {}) else architecture)
    identity = report.get("identities", {}).get(report_architecture)
    if not isinstance(identity, dict):
        raise ValueError("matched report lacks the selected full executable identity")
    if (identity.get("source_git_head") != (expected_reference or {}).get("source_git_head")
            or (component != "daily" and identity.get("horizon") != int(component[1:]))
            or identity.get("turnover_sha256") != expected_turnover_sha256
            or identity.get("native_AD_argnums") != [0, 1]
            or identity.get("training_updates") != 0
            or identity.get("precision") != "FP32_neural_FP64_physics"):
        raise ValueError("matched executable source/turnover/AD identity differs")
    measured_executable = report.get("cold_measurements", {}).get(report_architecture, {}).get("executable")
    if (not expected_executable or not measured_executable
            or Path(measured_executable).resolve() != Path(expected_executable).resolve()):
        raise ValueError("matched report does not bind the selected saved executable path")
    if architecture == CANDIDATE_ARCHITECTURE:
        candidate_identity = identity.get("parameter_provenance", {}).get("identity")
        if (expected_identity is None
                or json_identity(candidate_identity) != json_identity(expected_identity)
                or identity.get("architecture") != CANDIDATE_ARCHITECTURE
                or identity.get("parameter_count") != 66123713
                or identity.get("parameter_leaves") != 355):
            raise ValueError("matched candidate parameter identity differs")
        if (component == "daily" and expected_prepared_context_identity is not None
                and json_identity(identity.get("prepared_context_identity"))
                != json_identity(expected_prepared_context_identity)):
            raise ValueError("matched candidate prepared-context identity differs")
    else:
        if (expected_reference is None
                or json_identity(identity.get("method")) != json_identity(expected_reference.get("method"))
                or identity.get("architecture") != ORIGINAL_ARCHITECTURE
                or identity.get("parameter_count") != 64884481
                or identity.get("parameter_leaves") != 300):
            raise ValueError("matched original method identity differs")

    expected_shape = state_shape(component)
    expected_parameters = 66123713 if architecture == CANDIDATE_ARCHITECTURE else 64884481
    expected_leaves = 355 if architecture == CANDIDATE_ARCHITECTURE else 300
    calls = [row for row in report.get("results", ())
             if row.get("architecture") == report_architecture]
    if len(calls) < 2 or not any(row.get("label") == "cold-first" for row in calls):
        raise ValueError("matched report lacks repeated full-state AD validation calls")
    repeat_rows = 0
    state_repeat_failures = []
    for row in calls:
        validation = row.get("validation", {})
        if (validation.get("gradient_coordinates") != expected_parameters
                or validation.get("gradient_leaves") != expected_leaves
                or not validation.get("gradient_nonzero", 0) > 0
                or not np.isfinite(validation.get("gradient_norm", np.nan))
                or tuple(validation.get("initial_state_shape", ())) != expected_shape
                or validation.get("initial_state_finite") is not True
                or not validation.get("initial_state_nonzero", 0) > 0):
            raise ValueError("matched report lacks finite full parameter/state AD evidence")
        _hard_gate(validation)
        repeat = row.get("same_architecture_repeat")
        if repeat is not None:
            repeat_rows += 1
            if (repeat.get("all_parameter_gradients", {}).get("passed") is not True
                    or repeat.get("forward_and_gates", {}).get("passed") is not True
                    or repeat.get("loss_close") is False):
                raise ValueError("matched state-failure report also contains parameter/forward/loss repeat failures")
            state_repeat = repeat.get("all_initial_state_gradients", {})
            if state_repeat.get("passed") is False:
                state_repeat_failures.append({"label": row.get("label"),
                    "all_initial_state_gradients": state_repeat})
            elif state_repeat.get("passed") is not True:
                raise ValueError("matched report lacks explicit same-architecture state-repeat result")
    if repeat_rows < 1:
        raise ValueError("matched report lacks same-architecture parameter/forward/state repeat evidence")
    current_failures = report.get("numerical_failures", ())
    inherited_failures = report.get("inherited_source_strict_state_failures", ())
    current_state_failure_count = len(state_repeat_failures)
    current_pair_state_failure_count = report.get("current_pair_state_failure_count")
    if report.get("schema_version") == "shared_corrected_latent_reader_daily_pair_v1":
        if not isinstance(current_failures, list) or not isinstance(inherited_failures, list):
            raise ValueError("corrected daily failure provenance must be explicit arrays")
        if (not isinstance(current_pair_state_failure_count, int)
                or current_pair_state_failure_count != len(current_failures)
                or report.get("current_pair_state_parity_passed") is not (len(current_failures) == 0)):
            raise ValueError("corrected daily current state summary differs from saved failures")
        for failure in current_failures:
            comparison = failure.get("comparison", {})
            state = comparison.get("all_initial_state_gradients", {})
            if (failure.get("component", component) != component
                    or comparison.get("forward_and_gates", {}).get("passed") is not True
                    or comparison.get("all_parameter_gradients", {}).get("passed") is not True
                    or state.get("passed") is not False
                    or state.get("coordinates") != int(np.prod(expected_shape))
                    or not isinstance(state.get("failed"), int) or state["failed"] <= 0):
                raise ValueError("corrected daily report contains a non-state or incomplete current failure")
        for failure in inherited_failures:
            state = failure.get("all_initial_state_gradients", {})
            if (failure.get("architecture") != ORIGINAL_ARCHITECTURE
                    or failure.get("component") != "daily"
                    or state.get("passed") is not False
                    or state.get("coordinates") != 15416
                    or not isinstance(state.get("failed"), int) or state["failed"] <= 0):
                raise ValueError("corrected daily inherited source failure is not strict original state evidence")
        expected_status = "numerical_failed" if current_failures or inherited_failures else "passed"
        if report.get("status") != expected_status:
            raise ValueError("corrected daily global status differs from current/inherited state-only failures")
    elif ((report["status"] == "numerical_failed" and not state_repeat_failures)
            or (report["status"] == "passed" and state_repeat_failures)):
        raise ValueError("matched report status does not match the preserved state-repeat results")
    return {"report_status": report["status"], "scientific_acceptance": False,
            "architecture": architecture, "component": component,
            "identity": identity, "validated_calls": len(calls),
            "strict_state_repeat_admission": False, "performance_only": True,
            "numerical_failures": report.get("numerical_failures"),
            "state_repeat_failure_records": state_repeat_failures,
            "current_pair_state_parity_passed": report.get("current_pair_state_parity_passed"),
            "current_pair_state_failure_count": current_pair_state_failure_count,
            "inherited_source_strict_state_failures": inherited_failures}


def validate_corrected_leaf_daily_diagnostic(report, verification_report, sidecar, *,
                                              expected_reference, expected_turnover_sha256,
                                              expected_method_sha256, expected_executable,
                                              expected_prepared_context_identity):
    """Admit the original corrected leaf daily graph's distinct report schema."""
    identity = report.get("identity", {})
    context = identity.get("prepared_context_identity", {})
    method = context.get("method")
    method_sha256 = hashlib.sha256(json.dumps(
        method, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    expected_context = expected_prepared_context_identity
    observed_context = {
        "method_sha256": method_sha256,
        "artifact_id": context.get("artifact_id"),
        "daily_rows": context.get("daily_rows"),
        "initial_checkpoint": context.get("initial_checkpoint"),
    }
    if (report.get("schema_version") != "shared_leaf_corrected_full_state_cuda_v1"
            or report.get("status") != "completed"
            or report.get("scientific_acceptance") is not False
            or report.get("adoption") is not False
            or report.get("finite_and_physical_passed") is not True
            or report.get("meaningful_gradients_passed") is not True
            or report.get("observed_peak_memory_passed") is not True
            or report.get("same_input_strict_parity") is not False
            or identity.get("source_git_head") != expected_reference.get("source_git_head")
            or identity.get("turnover_sha256") != expected_turnover_sha256
            or identity.get("native_AD_argnums") != [0, 1]
            or identity.get("parameters") != 64884481
            or identity.get("parameter_leaves") != 300
            or identity.get("precision") != "FP32_neural_FP64_physical"
            or identity.get("loss") != "unchanged_public_joint_daily_mean4"
            or json_identity(method) != json_identity(expected_reference.get("method"))
            or method_sha256 != expected_method_sha256
            or not isinstance(expected_context, dict)
            or set(expected_context) != set(observed_context)
            or json_identity(observed_context) != json_identity(expected_context)
            or expected_context.get("method_sha256") != expected_method_sha256
            or expected_context.get("artifact_id") != expected_reference.get("method", {}).get(
                "training_plan", {}).get("training_artifact_ids", [None])[0]
            or expected_context.get("daily_rows") != 4
            or sidecar.get("component") != "daily"
            or json_identity(sidecar.get("identity")) != json_identity(identity)):
        raise ValueError("corrected original leaf daily identity/schema differs")
    if Path(expected_executable).name != "leaf-full-state-cuda-20261005-full-state.executable.pkl":
        raise ValueError("corrected original leaf daily executable path differs")
    results = report.get("results", ())
    if len(results) != 4:
        raise ValueError("corrected original leaf daily report must contain four full-state calls")
    for row in results:
        validation = row.get("validation", {})
        shapes = validation.get("state_diagnostic_group_shapes", {})
        expected_shapes, predicted_shapes = shapes.get("expected"), shapes.get("predicted")
        if (validation.get("finite") is not True
                or validation.get("physical_hard_counts_zero") is not True
                or validation.get("parameter_gradient_nonzero", 0) <= 0
                or validation.get("initial_state_gradient_nonzero", 0) <= 0
                or any(value != 0 for value in validation.get("hard_counts", {}).values())
                or not expected_shapes or expected_shapes != predicted_shapes
                or expected_shapes[-1] != [4, 3854]):
            raise ValueError("corrected original leaf daily call has incomplete finite/physical/state evidence")
    if (verification_report.get("schema_version") != "shared_leaf_cuda_preserved_result_inspection_v1"
            or verification_report.get("status") != "inspected"
            or verification_report.get("scientific_acceptance") is not False
            or verification_report.get("GPU_executed") is not False
            or verification_report.get("original_forward_parameter_passed") is not True
            or verification_report.get("repeat_forward_parameter_passed") is not True
            or verification_report.get("repeat_all_state_passed") is not False
            or verification_report.get("corrected_finite_physical_reported") is not True):
        raise ValueError("corrected original leaf daily verification report differs")
    repeats = verification_report.get("repeat_comparisons", {})
    regressions = verification_report.get("original_regression", {})
    state_failures = []
    regression_evidence = []
    for schedule in ("old", "new"):
        evidence = repeats.get(schedule, {})
        if (evidence.get("all_parameter_gradients", {}).get("passed") is not True
                or evidence.get("forward_and_gates", {}).get("passed") is not True):
            raise ValueError("corrected original leaf daily parameter/forward repeat failed")
        state = evidence.get("all_initial_state_gradients", {})
        if (state.get("passed") is not False or state.get("coordinates") != 15416
                or not isinstance(state.get("failed"), int) or state["failed"] <= 0):
            raise ValueError("corrected original leaf daily state repeat result is unavailable")
        state_failures.append({"schedule": schedule, "all_initial_state_gradients": state})
        regression = regressions.get(schedule, {}).get("state_gradient_changes")
        if (not isinstance(regression, dict) or regression.get("passed") is not False
                or not isinstance(regression.get("failed"), int)):
            raise ValueError("corrected-versus-original regression evidence is unavailable")
        regression_evidence.append({"schedule": schedule, "state_gradient_changes": regression})
    if verification_report.get("source_corrected_report") != str(expected_executable).replace(
            ".executable.pkl", ".json"):
        raise ValueError("leaf daily verification report is not bound to corrected report path")
    return {"report_status": "numerical_failed", "source_report_status": report["status"],
            "scientific_acceptance": False, "architecture": ORIGINAL_ARCHITECTURE,
            "component": "daily", "identity": identity, "validated_calls": len(report.get("results", ())),
            "strict_state_repeat_admission": False, "performance_only": True,
            "state_repeat_failure_records": state_failures,
            "source_corrected_vs_original_regression": regression_evidence}


def validate_full_state_abi(jax, serialized, executable, arguments, *, component):
    """Check full explicit state arguments and every serialized shape/dtype."""
    expected_length = 3 if component == "daily" else 6
    if len(arguments) != expected_length or tuple(arguments[1].shape) != state_shape(component):
        raise ValueError("full-state argument count/state shape differs")
    supplied = (tuple(arguments), {})
    if not isinstance(serialized, tuple) or len(serialized) != 3:
        raise ValueError("serialized executable must be the JAX serialization triple")
    if serialized[1] != jax.tree_util.tree_structure(supplied):
        raise ValueError("serialized full-state input treedef differs")
    infos, tree = jax.tree_util.tree_flatten(executable.args_info)
    values, other = jax.tree_util.tree_flatten(supplied)
    if tree != other:
        raise ValueError("full-state args_info tree differs")
    for info, value in zip(infos, values, strict=True):
        if tuple(info._aval.shape) != tuple(value.shape) or np.dtype(info._aval.dtype) != np.dtype(value.dtype):
            raise ValueError("full-state input shape/dtype differs")


def load_full_state_executable(jax, path, arguments, *, component, admission,
                               architecture, expected_identity=None, expected_reference=None,
                               admission_mode=SCIENTIFIC_ADMISSION,
                               expected_turnover_sha256=None,
                               corrected_daily_verification=None,
                               expected_method_sha256=None,
                               expected_prepared_context_identity=None,
                               deserialize=None, memory_sizes=None, compile_guard=None,
                               memory_limit_bytes=24 * 2**30):
    """Load an explicitly admitted full binary; dependencies support CPU tests.

    Pickle paths are trusted project artifacts. No compilation fallback exists.
    Candidate sidecars are mandatory; original identity comes from its accepted
    report, whose executable reference must match the selected filename.
    """
    path = Path(path)
    if isinstance(admission, (str, Path)):
        admission = json.loads(Path(admission).read_text())
    diagnostic_view = None
    if admission_mode == CORRECTED_PERFORMANCE_DIAGNOSTIC:
        sidecar = json.loads(path.with_suffix(path.suffix + ".identity.json").read_text())
        if (component == "daily" and architecture == ORIGINAL_ARCHITECTURE
                and admission.get("schema_version") == "shared_leaf_corrected_full_state_cuda_v1"):
            if corrected_daily_verification is None or not expected_method_sha256:
                raise ValueError("corrected original daily verification identity required")
            if isinstance(corrected_daily_verification, (str, Path)):
                corrected_daily_verification = json.loads(Path(corrected_daily_verification).read_text())
            diagnostic_view = validate_corrected_leaf_daily_diagnostic(
                admission, corrected_daily_verification, sidecar,
                expected_reference=expected_reference,
                expected_turnover_sha256=expected_turnover_sha256,
                expected_method_sha256=expected_method_sha256,
                expected_executable=path,
                expected_prepared_context_identity=expected_prepared_context_identity)
            declared_path = str(path)
        else:
            diagnostic_view = validate_corrected_performance_diagnostic(
                admission, component, architecture, expected_identity=expected_identity,
                expected_reference=expected_reference,
                expected_turnover_sha256=expected_turnover_sha256,
                expected_executable=path,
                expected_prepared_context_identity=(expected_prepared_context_identity
                    if architecture == CANDIDATE_ARCHITECTURE and component == "daily" else None))
            if (sidecar.get("component") != component
                    or json_identity(sidecar.get("identity"))
                    != json_identity(diagnostic_view["identity"])):
                raise ValueError("corrected executable sidecar identity/component differs")
            declared_path = admission["cold_measurements"][architecture]["executable"]
    elif admission_mode == SCIENTIFIC_ADMISSION and architecture == CANDIDATE_ARCHITECTURE:
        if expected_identity is None:
            raise ValueError("candidate expected identity required")
        validate_candidate_admission(admission, component, expected_identity)
        sidecar = json.loads(path.with_suffix(path.suffix + ".identity.json").read_text())
        if sidecar != json_identity({"identity": expected_identity, "component": component}):
            raise ValueError("candidate executable sidecar identity/component differs")
        declared_path = admission.get("executable")
    elif admission_mode == SCIENTIFIC_ADMISSION and architecture == ORIGINAL_ARCHITECTURE:
        if expected_reference is None:
            raise ValueError("original expected reference required")
        validate_original_admission(admission, component, expected_reference)
        declared_path = admission.get("reference_executable", admission.get("executable"))
    else:
        raise ValueError("explicit supported architecture and admission mode required")
    # Reports may be mirrored locally, retaining their original remote paths.
    if not declared_path or Path(declared_path).name != path.name:
        raise ValueError("admitted executable filename differs")
    if deserialize is None:
        from jax.experimental.serialize_executable import deserialize_and_load
        deserialize = deserialize_and_load
    if memory_sizes is None:
        from scripts.hpc.run_shared_transformer_admission import memory_sizes
    with compile_guard() if compile_guard is not None else nullcontext():
        with path.open("rb") as stream:
            serialized = pickle.load(stream)
        executable = deserialize(*serialized)
        validate_full_state_abi(jax, serialized, executable, arguments, component=component)
    memory = memory_sizes(executable)
    size = memory.get("estimated_live_bytes")
    if not isinstance(size, (int, float)) or not np.isfinite(size) or not 0 <= size <= memory_limit_bytes:
        raise MemoryError("full-state executable memory unavailable/exceeds limit")
    return executable, {"architecture": architecture, "component": component,
                        "identity": (diagnostic_view["identity"] if diagnostic_view else admission["identity"]),
                        "admission_mode": admission_mode,
                        "report_status": (diagnostic_view["report_status"] if diagnostic_view else "passed"),
                        "scientific_acceptance": False, "compiled_memory": memory,
                        "full_state_abi_validated": True, "model_compiles": 0}


class FullStateADEvaluator:
    """Canonical parameter-only return view of a full parameter/state AD binary.

    State transfer/validation happens within each call, hence inside existing
    branch wall timers. Parameter gradients remain on device. Records contain
    full small state arrays, bounded to eight daily calls or one recurrent call
    per update; callers must take/reset between complete updates and warmups.
    """
    def __init__(self, jax, executable, component, *, max_records=None, compile_guard=None):
        self.jax = jax
        self.executable = executable
        self.component = component
        self.compile_guard = compile_guard
        self.expected_shape = state_shape(component)
        self.max_records = (8 if component == "daily" else 1) if max_records is None else max_records
        if not isinstance(self.max_records, int) or self.max_records <= 0:
            raise ValueError("positive bounded record count required")
        self._records = []

    def reset(self):
        self._records.clear()

    def take_records(self):
        records = tuple(self._records)
        self.reset()
        return records

    def __call__(self, parameters, *case):
        if len(self._records) >= self.max_records:
            raise ValueError("full-state records must be consumed/reset between updates")
        if self.component == "daily":
            if len(case) != 1:
                raise ValueError("daily canonical call requires samples")
            arguments = (parameters, case[0].inputs.continuous_state, case[0])
        else:
            if len(case) != 5:
                raise ValueError("recurrent canonical call requires the full five-element case")
            arguments = (parameters, *case)
        initial_state = arguments[1]
        if tuple(initial_state.shape) != self.expected_shape:
            raise ValueError("full-state input shape differs")
        with self.compile_guard() if self.compile_guard is not None else nullcontext():
            result = self.executable(*arguments)
        if (not isinstance(result, (tuple, list)) or len(result) != 2
                or not isinstance(result[0], (tuple, list)) or len(result[0]) != 2
                or not isinstance(result[1], (tuple, list)) or len(result[1]) != 2):
            raise ValueError("executable did not return full parameter/state AD")
        parameter_gradient, state_gradient = result[1]
        for value in self.jax.tree_util.tree_leaves(self.jax.device_get(result[0])):
            if not np.all(np.isfinite(np.asarray(value))):
                raise ValueError("full-state loss/auxiliary contains nonfinite values")
        # Tree equality distinguishes a two-leaf parameter-only gradient from
        # an actual (parameter tree, state array) result without downloading it.
        if self.jax.tree_util.tree_structure(parameter_gradient) != self.jax.tree_util.tree_structure(parameters):
            raise ValueError("full-state parameter gradient tree differs")
        for parameter, gradient in zip(self.jax.tree_util.tree_leaves(parameters),
                                       self.jax.tree_util.tree_leaves(parameter_gradient), strict=True):
            if tuple(parameter.shape) != tuple(gradient.shape) or np.dtype(parameter.dtype) != np.dtype(gradient.dtype):
                raise ValueError("full-state parameter gradient shape/dtype differs")
        state = np.asarray(self.jax.device_get(state_gradient))
        if (state.shape != self.expected_shape or state.dtype != np.dtype(initial_state.dtype)
                or not np.all(np.isfinite(state))):
            raise ValueError("full initial-state gradient shape/dtype/finite gate failed")
        self._records.append({"component": self.component, "call_index": len(self._records),
                              "initial_state_gradient": state.copy()})
        return result[0], parameter_gradient
