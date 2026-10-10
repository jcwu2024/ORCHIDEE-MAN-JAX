"""Prepare or run the bounded same-precision T7 B1/B4 state-JVP diagnostic."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

try:
    from scripts.hpc import probe_shared_latent_reader_T7_pair as pair
    from scripts.hpc import shared_latent_T7_batch_ad as batch_ad
    from scripts.hpc import shared_latent_T7_batch_assets as batch_assets
    from scripts.hpc import probe_shared_latent_T7_batch as batch_probe
    from scripts.hpc import shared_latent_T7_state_jvp as contract
except ImportError:
    import probe_shared_latent_reader_T7_pair as pair
    import shared_latent_T7_batch_ad as batch_ad
    import shared_latent_T7_batch_assets as batch_assets
    import probe_shared_latent_T7_batch as batch_probe
    import shared_latent_T7_state_jvp as contract


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("cpu-readiness", "execute"), required=True)
    p.add_argument("--route-mode", choices=("both", "B4-only", "B1-completion"), default="both")
    p.add_argument("--repair-record", type=Path)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--horizon", choices=(7,), type=int, default=7)
    for name in ("manifest", "prepared-context", "parameters", "field-adapter", "backbone-adapter",
                 "transformer-module", "latent-module", "turnover-overlay", "runtime-preflight",
                 "training-prepared-manifest", "dataset-manifest", "b1-executable", "b4-executable",
                 "paid-pair-envelope", "state-input-metadata", "approval-report", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--b4-jvp-executable", type=Path)
    p.add_argument("--wall-seconds", type=int, default=3300)
    p.add_argument("--gpu-seconds-cap", type=int, default=3600)
    p.add_argument("--lower-seconds", type=int, default=180)
    p.add_argument("--backend-seconds", type=int, default=900)
    p.add_argument("--route-seconds", type=int, default=1500)
    p.add_argument("--load-seconds", type=int, default=300)
    p.add_argument("--force-compilation-parallelism", choices=(0, 4), type=int, default=4)
    p.add_argument("--turnover-base-sha256", default=pair.BASE_SHA)
    p.add_argument("--turnover-sha256", default=pair.CORRECTED_SHA)
    return p


def _candidate_objective(args, owner, transformer, context, candidate_head, patches):
    from research.daily_coarse_graining.daily_operator import shared_process_head
    from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_recurrent_objective
    from jax_orchidee.stomate import carbon_kernels
    patches.enter_context(pair.install_turnover_overlay(carbon_kernels, args.turnover_overlay, pair.BASE_SHA))
    assembler = owner.load_backbone_assembler(args.backbone_adapter)
    patches.enter_context(pair.candidate_backbone_patches(
        shared_process_head, transformer.shared_transformer_raw_outputs, assembler))
    _, objective = make_shared_recurrent_objective(candidate_head, context["template"],
        context["state_scales"], context["science_scales"])
    return objective


def route_functions(jax, objective, *, retain_unscaled_b1=False):
    """JVPs keep every public argument dynamic and return primal aux bridges."""
    mean = batch_ad.make_batch_objective(jax, objective)

    def b1(parameters, state, discrete, days, targets, contexts, direction):
        def one(initial):
            loss, auxiliary = objective(parameters, initial, discrete, days, targets, contexts)
            return loss, auxiliary
        primal, tangent, auxiliary = jax.jvp(one, (state,),
            (direction[contract.SAMPLE_INDEX],), has_aux=True)
        if retain_unscaled_b1:
            return (primal / 4, auxiliary), (tangent, tangent / 4)
        return (primal / 4, auxiliary), tangent / 4

    def b4(parameters, states, discrete, days, targets, contexts, direction):
        def many(initial_states):
            return mean(parameters, initial_states, discrete, days, targets, contexts)
        primal, tangent, auxiliary = jax.jvp(many, (states,), (direction,), has_aux=True)
        return (primal, auxiliary), tangent

    return b1, b4


def primal_bridge_routes(route):
    """Only compare primal/auxiliary trees produced by the same route."""
    if route not in ("B1", "B4"):
        raise ValueError("known JVP route required")
    return (route,)


def _approval(args):
    approval = json.loads(args.approval_report.read_text(encoding="utf-8"))
    route_mode = getattr(args, "route_mode", "both")
    recovery = route_mode == "B4-only"
    completion = route_mode == "B1-completion"
    if completion:
        expected = {"schema_version": "latent_T7_B1_state_jvp_completion_approval_v1",
                    "funding_scope": "latent_T7_B1_state_jvp_completion_20261009",
                    "gpu_count": 1, "node": "gln01", "limit_gpu_seconds": 3600,
                    "new_B1_JVP_compile_attempts": 1, "new_B4_JVP_compile_attempts": 0,
                    "optimizer_update_cap": 0, "probe_wall_seconds": 3300,
                    "protected_exit_reserve_seconds": 300, "B1_lower_seconds": 180,
                    "B1_backend_seconds": 1800, "route_seconds": 2400,
                    "saved_executable_load_seconds_each": 300, "saved_executable_load_count": 3,
                    "jvp_calls": 32, "saved_vjp_calls": 4, "optimizer_updates": 0,
                    "worst_case_cost_cny": 0.0, "paid_fallback": False,
                    "approval_required_before_gpu_execution": True,
                    "entry_script": "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-B1-state-jvp-completion-20261009/probe_shared_latent_T7_state_jvp.py",
                    "working_directory": "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1",
                    "output_report": str(args.output)}
        approval = json.loads(args.approval_report.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if type(approval.get(key)) is not type(value) or approval.get(key) != value:
                raise ValueError("B1 completion approval scope differs: " + key)
        if approval.get("approved") is not True:
            raise PermissionError("GPU execution requires a separately approved B1 completion allocation")
        if approval.get("approval_origin") != "explicit_human_user" or not str(approval.get("approval_note", "")).strip():
            raise PermissionError("GPU execution requires an explicit approval note")
        return approval
    approved_output = args.output.parent / "T7-state-jvp-20261009.result.json" if recovery else args.output
    expected = {"schema_version": "latent_T7_state_jvp_approval_v1",
                "funding_scope": "latent_T7_state_jvp_20261009", "gpu_count": 1,
                "node": "gln01", "limit_gpu_seconds": 3600, "model_compile_cap_per_route": 1,
                "optimizer_update_cap": 0,
                "entry_script": "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-state-jvp-20261009/probe_shared_latent_T7_state_jvp.py",
                "working_directory": "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1",
                "output_report": str(approved_output)}
    for key, value in expected.items():
        if type(approval.get(key)) is not type(value) or approval.get(key) != value:
            raise ValueError("JVP approval scope differs: " + key)
    if approval.get("approved") is not True:
        raise PermissionError("GPU execution requires a separately approved allocation")
    if approval.get("approval_origin") != "explicit_human_user" or not str(approval.get("approval_note", "")).strip():
        raise PermissionError("GPU execution requires an explicit approval note")
    repair_path = getattr(args, "repair_record", None)
    if recovery:
        if repair_path is None:
            raise PermissionError("B4-only recovery requires a separately bound repair record")
        repair = json.loads(repair_path.read_text(encoding="utf-8"))
        expected_repair = {"schema_version": "latent_T7_state_jvp_recovery_budget_v1",
            "route_mode": "B4-only", "recovery_output": str(args.output),
            "original_approval_sha256": hashlib.sha256(args.approval_report.read_bytes()).hexdigest(),
            "approved_total_grant_gpu_seconds": 3600, "previously_charged_gpu_seconds": 1149.853161,
            "remaining_before_recovery_gpu_seconds": 2450.146839,
            "reserved_recovery_gpu_seconds": 2400, "protected_remainder_gpu_seconds": 50.146839,
            "lower_seconds": 180, "backend_seconds": 1500, "route_seconds": 1800,
            "saved_executable_load_seconds_each": 300, "reuse_runtime_preflight": True,
            "JVP_calls": 16, "saved_VJP_calls": 4, "optimizer_updates": 0}
        for key, value in expected_repair.items():
            if type(repair.get(key)) is not type(value) or repair.get(key) != value:
                raise ValueError("recovery repair record differs: " + key)
    elif repair_path is not None:
        raise ValueError("repair record is only valid for B4-only recovery")
    return approval


def _save(owner, path, value):
    if path.exists():
        raise FileExistsError(path)
    owner.atomic_pickle(path, value)


def compare_primal_bridge(jax, route, jvp_primal_aux, saved_result):
    saved_loss, saved_aux = saved_result
    jvp_loss, jvp_aux = jvp_primal_aux
    if route == "B1":
        saved_loss = saved_loss / 4
    actual = (jvp_loss, jvp_aux)
    reference = (saved_loss, saved_aux)
    if jax.tree_util.tree_structure(actual) != jax.tree_util.tree_structure(reference):
        raise ValueError(route + " public primal/auxiliary pytree differs from saved VJP")
    leaves_a = jax.tree_util.tree_leaves(actual)
    leaves_r = jax.tree_util.tree_leaves(reference)
    records = []
    for index, (a, r) in enumerate(zip(leaves_a, leaves_r, strict=True)):
        aa, rr = np.asarray(a), np.asarray(r)
        if aa.shape != rr.shape or aa.dtype != rr.dtype or not np.all(np.isfinite(aa)) or not np.all(np.isfinite(rr)):
            raise ValueError(route + " JVP primal/auxiliary shape, dtype, or finite gate failed")
        tolerance = contract.ATOL + contract.RTOL * np.abs(rr)
        failed = np.abs(aa - rr) > tolerance
        records.append({"leaf": index, "shape": list(aa.shape), "dtype": str(aa.dtype),
            "failed_coordinates": int(np.count_nonzero(failed)),
            "max_absolute_error": float(np.max(np.abs(aa - rr), initial=0.0)),
            "passed": bool(not np.any(failed))})
    hard = jvp_aux.get("hard") if isinstance(jvp_aux, dict) else None
    if not isinstance(hard, dict) or not hard:
        raise ValueError(route + " JVP primal is missing public physical/mask hard gates")
    hard_checks = {str(name): {"zero": bool(np.all(np.asarray(value) == 0)),
                               "values": np.asarray(value).tolist()}
                   for name, value in hard.items()}
    return {"passed": all(row["passed"] for row in records)
                       and all(row["zero"] for row in hard_checks.values()),
            "rtol": contract.RTOL, "atol": contract.ATOL,
            "leaves": records, "hard_physical_mask_gates": hard_checks}


def validate_serialized_call_abi(jax, serialized, executable, arguments, route):
    """Validate a loaded JVP executable's exact dynamic argument ABI."""
    if not isinstance(serialized, tuple) or len(serialized) != 3:
        raise ValueError(route + " serialized executable triple required")
    supplied = (tuple(arguments), {})
    if serialized[1] != jax.tree_util.tree_structure(supplied):
        raise ValueError(route + " serialized argument treedef differs")
    infos, info_tree = jax.tree_util.tree_flatten(executable.args_info)
    values, value_tree = jax.tree_util.tree_flatten(supplied)
    if info_tree != value_tree:
        raise ValueError(route + " loaded args_info treedef differs")
    for info, value in zip(infos, values, strict=True):
        if (tuple(info._aval.shape) != tuple(value.shape)
                or np.dtype(info._aval.dtype) != np.dtype(value.dtype)):
            raise ValueError(route + " loaded argument shape/dtype differs")
    return {"route": route, "argument_count": len(arguments),
            "argument_leaf_count": len(values),
            "argument_leaf_shapes": [list(value.shape) for value in values],
            "argument_leaf_dtypes": [str(value.dtype) for value in values]}


def cpu_metadata_readiness(args, owner):
    """Bind saved graph identities and audited inputs without importing JAX."""
    approval = json.loads(args.approval_report.read_text(encoding="utf-8"))
    if approval.get("approved") is not False:
        raise ValueError("CPU preparation record must retain approved=false")
    metadata = json.loads(args.state_input_metadata.read_text(encoding="utf-8"))
    directions = contract.direction_matrix(dtype=np.float64)
    input_record = contract.validate_state_input_metadata(metadata, directions)
    b1_path = args.b1_executable.with_suffix(args.b1_executable.suffix + ".identity.json")
    b4_path = args.b4_executable.with_suffix(args.b4_executable.suffix + ".identity.json")
    b1_sidecar = json.loads(b1_path.read_text(encoding="utf-8"))
    b4_sidecar = json.loads(b4_path.read_text(encoding="utf-8"))
    identity = dict(b1_sidecar.get("identity", {}))
    base_identity = dict(identity)
    base_identity.pop("device_kind", None)
    b4_identity = b4_sidecar.get("identity", {})
    if (b1_sidecar.get("component") != "T7" or b4_sidecar.get("component") != "B4_T7"
            or b4_identity.get("saved_B1_mathematical_identity") != base_identity):
        raise ValueError("saved B1/B4 executable identities are not the same candidate graph contract")
    if not args.b1_executable.is_file() or not args.b4_executable.is_file():
        raise FileNotFoundError("both saved VJP executable binaries are required")
    saved_b4_jvp = None
    saved_b4_jvp_identity_sha256 = None
    if args.route_mode == "B1-completion":
        if args.b4_jvp_executable is None or not args.b4_jvp_executable.is_file():
            raise FileNotFoundError("saved B4 JVP executable is required for B1 completion")
        saved_b4_jvp_path = args.b4_jvp_executable.with_suffix(
            args.b4_jvp_executable.suffix + ".identity.json")
        saved_b4_jvp = json.loads(saved_b4_jvp_path.read_text(encoding="utf-8"))
        saved_b4_jvp_identity_sha256 = pair.sha(saved_b4_jvp_path)
        if (saved_b4_jvp.get("route") != "B4"
                or saved_b4_jvp.get("identity") != b4_identity
                or saved_b4_jvp.get("compiler_options") != b4_identity.get(
                    "saved_B1_mathematical_identity", {}).get("compiler_options")):
            raise ValueError("saved B4 JVP graph identity/compiler options differ from the saved B4 VJP")
    if identity.get("source_git_head") != "c599236bcf57b745e9bafed678298010f4c3b58f":
        raise ValueError("candidate binary source checkout differs")
    candidate = identity.get("parameter_provenance", {})
    if (candidate.get("initialization") != "fresh_public_whole_tree"
            or candidate.get("sha256") != pair.sha(args.parameters)
            or Path(candidate.get("artifact", "")).name != args.parameters.name):
        raise ValueError("exact fresh candidate parameter artifact differs")
    module_hashes = identity.get("neural_module_sha256", {})
    if (module_hashes.get("shared_transformer.py") != pair.sha(args.transformer_module)
            or module_hashes.get("shared_latent_reader.py") != pair.sha(args.latent_module)):
        raise ValueError("candidate neural source hashes differ")
    if (identity.get("prepared_context_sha256") != pair.sha(args.prepared_context)
            or metadata.get("identities", {}).get("prepared_context_sha256") != pair.sha(args.prepared_context)):
        raise ValueError("saved prepared-context identity differs")
    if (identity.get("turnover_base_sha256") != pair.BASE_SHA
            or identity.get("turnover_sha256") != pair.CORRECTED_SHA
            or pair.sha(args.turnover_overlay) != pair.CORRECTED_SHA):
        raise ValueError("accepted guarded turnover source identity differs")
    sampling = b4_identity.get("sampling", ())
    artifact_ids = [row.get("artifact_id") for row in sampling]
    if artifact_ids != [742, 275, 482, 299] or any(
            row.get("spatial_split") != "train" or row.get("temporal_split") != "train" for row in sampling):
        raise ValueError("saved B4 input identity is not the exact four TRAIN windows")
    envelope = json.loads(args.paid_pair_envelope.read_text(encoding="utf-8"))
    contract.paid_pair_envelope_report({"B1": np.zeros(4), "B4": np.zeros(4)}, envelope)
    return {"status": "CPU_readiness_passed", "GPU_executed": False, "full_model_trace": False,
        "JAX_numeric_calls": 0, "source_git_head": identity["source_git_head"],
        "saved_B1_identity_sha256": pair.sha(b1_path), "saved_B4_identity_sha256": pair.sha(b4_path),
        "saved_B1_executable_sha256": pair.sha(args.b1_executable),
        "saved_B4_executable_sha256": pair.sha(args.b4_executable),
        "saved_B4_JVP_executable_sha256": pair.sha(args.b4_jvp_executable) if saved_b4_jvp else None,
        "saved_B4_JVP_identity_sha256": saved_b4_jvp_identity_sha256,
        "mathematical_identity": b4_identity, "selected_artifact_ids": artifact_ids,
        "sample3_state": input_record,
        "directions": {"shape": list(directions.shape), "dtype": str(directions.dtype),
                       "sample3_only": True, "coordinates": list(contract.STATE_COORDINATES)},
        "paid_pair_envelope_sha256": pair.sha(args.paid_pair_envelope),
        "model_compiles": {"B1_JVP": 0, "B4_JVP": 0}, "optimizer_updates": 0,
        "saved_B4_JVP_reused_without_recompile": saved_b4_jvp is not None,
        "adoption": False, "scientific_acceptance": False}


def run(args):
    cpu = args.stage == "cpu-readiness"
    if (args.horizon != 7 or args.turnover_base_sha256 != pair.BASE_SHA
            or args.turnover_sha256 != pair.CORRECTED_SHA):
        raise ValueError("exact corrected T7 protocol required")
    if args.output.exists():
        raise FileExistsError(args.output)
    calls = 16 if args.route_mode == "B4-only" else 32
    caps = contract.validate_occupation_plan(total_seconds=args.gpu_seconds_cap,
        lower_seconds=args.lower_seconds, backend_seconds=args.backend_seconds,
        route_seconds=args.route_seconds, load_seconds=args.load_seconds, calls=calls,
        route_mode=args.route_mode)
    if args.wall_seconds > 3300:
        raise ValueError("probe wall cap must preserve the 300-second exit reserve")
    if args.plan_only:
        return {"status": "plan_only", "route_mode": args.route_mode,
            "GPU_executed": False, "GPU_occupation_started": False,
            "gpu_backend_started": False, "backend_compile_attempted": False,
            "real_model_evaluated": False,
            "route_measurement_status": {"B1_JVP": "unmeasured" if args.route_mode == "B4-only" else "pending",
                "B4_JVP": "planned" if args.route_mode in ("B4-only", "B1-completion") else "pending"},
            "model_compiles": {"B1_JVP": 0, "B4_JVP": 0}, "JVP_calls": 0,
            "saved_VJP_calls": 0, "planned_model_compiles": {
                "B1_JVP": int(args.route_mode in ("both", "B1-completion")),
                "B4_JVP": int(args.route_mode in ("both", "B4-only"))},
            "planned_JVP_calls": calls, "planned_saved_VJP_calls": 4,
            "saved_executable_load_count": caps["saved_executable_load_count"],
            "saved_executables_to_load": ["B1_VJP", "B4_VJP", "B4_JVP"]
                if args.route_mode == "B1-completion" else ["B1_VJP", "B4_VJP"],
            "saved_B4_JVP_reused_without_recompile": args.route_mode == "B1-completion",
            "optimizer_updates": 0, "occupation_caps": caps}
    if cpu:
        os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        approval = json.loads(args.approval_report.read_text(encoding="utf-8"))
        if approval.get("approved") is not False:
            raise ValueError("CPU readiness report must retain approved=false")
    else:
        _approval(args)
        if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
            raise ValueError("one approved allocated GPU required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pair.adjacent("training_artifact_io")
    journal = pair.Journal(args.output, args.wall_seconds)
    journal.report.update(schema_version="shared_latent_T7_state_jvp_v1", requested_stage=args.stage,
        GPU_executed=False, gpu_count=0 if cpu else 1, model_compiles={"B1_JVP": 0, "B4_JVP": 0},
        GPU_occupation_started=False, gpu_backend_started=False,
        backend_compile_attempted=False, real_model_evaluated=False,
        GPU_executed_semantics="GPU backend use; see real_model_evaluated for model execution",
        GPU_occupation_accounting="shared_gpu_budget occupation ledger",
        saved_VJP_calls=0, JVP_calls=0, optimizer_updates=0, training_updates=0,
        route_mode=args.route_mode,
        route_measurement_status={"B1_JVP": "unmeasured" if args.route_mode == "B4-only" else "pending",
                                 "B4_JVP": "pending"},
        adoption=False, scientific_acceptance=False, rtol=contract.RTOL, atol=contract.ATOL,
        occupation_caps=caps, events=[], comparisons=[])
    try:
        if cpu:
            readiness = cpu_metadata_readiness(args, pair.adjacent("training_artifact_io"))
            journal.report.update(readiness)
            journal.emit("CPU_metadata_inputs_and_saved_ABIs_verified_no_JAX_trace")
            return journal.report
        if not cpu:
            pair.adjacent("orcjax_gpu_runtime").initialize_gpu_backend()
            journal.report.update(GPU_executed=True, GPU_occupation_started=True,
                                  gpu_backend_started=True)
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        devices = jax.devices()
        if cpu and jax.default_backend() != "cpu":
            raise ValueError("CPU readiness requires CPU backend")
        if not cpu and (len(devices) != 1 or devices[0].platform != "gpu" or "V100" not in devices[0].device_kind):
            raise ValueError("one approved free-node V100 required")
        if not cpu:
            journal.report["runtime_preflight"] = pair.adjacent("probe_shared_union_micro4").validate_runtime_report(
                args.runtime_preflight, jax.__version__)

        source, owner, transformer, context, head, _, trees, identities = pair.assets(jax, args)
        parameters = trees["candidate"]
        del trees
        manifest = pair.read_manifest(args)
        selector = pair.adjacent("shared_latent_T7_batch_assets")
        with jax.default_device(jax.devices("cpu")[0]):
            selected = selector.load_training_T7_cases(jax,
                prepared_manifest=args.training_prepared_manifest, dataset_manifest=args.dataset_manifest,
                input_bundle=manifest["input_bundle"], head=head,
                method=identities["candidate"]["method"], count=4)
        cases, stacked = selected["cases"], selected["stacked"]
        if list(selected["selected_artifact_ids"]) != [742, 275, 482, 299]:
            raise ValueError("exact four TRAIN artifact IDs 742/275/482/299 required")
        if any(row["spatial_split"] != "train" or row["temporal_split"] != "train"
               for row in selected["window_metadata"]):
            raise ValueError("TRAIN-only recurrent inputs required")
        identity = batch_probe.mathematical_identity(jax, identities["candidate"], selected)
        b1_sidecar_path = args.b1_executable.with_suffix(args.b1_executable.suffix + ".identity.json")
        b1_sidecar = json.loads(b1_sidecar_path.read_text(encoding="utf-8"))
        b1_identity = dict(b1_sidecar.get("identity", {}))
        b1_identity.pop("device_kind", None)
        expected_base = dict(identities["candidate"])
        expected_base.pop("device_kind", None)
        if b1_sidecar.get("component") != "T7" or b1_identity != expected_base:
            raise ValueError("saved B1 full-state candidate identity differs")
        b4_sidecar_path = args.b4_executable.with_suffix(args.b4_executable.suffix + ".identity.json")
        b4_sidecar = json.loads(b4_sidecar_path.read_text(encoding="utf-8"))
        if b4_sidecar.get("identity") != identity or b4_sidecar.get("component") != "B4_T7":
            raise ValueError("saved B4 mean-objective mathematical identity differs")
        if args.route_mode == "B1-completion":
            if args.b4_jvp_executable is None or not args.b4_jvp_executable.is_file():
                raise FileNotFoundError("saved B4 JVP executable is required for B1 completion")
            b4_jvp_sidecar_path = args.b4_jvp_executable.with_suffix(
                args.b4_jvp_executable.suffix + ".identity.json")
            b4_jvp_sidecar = json.loads(b4_jvp_sidecar_path.read_text(encoding="utf-8"))
            if (b4_jvp_sidecar.get("route") != "B4"
                    or b4_jvp_sidecar.get("identity") != b4_sidecar.get("identity")
                    or b4_jvp_sidecar.get("compiler_options") != identity.get(
                        "saved_B1_mathematical_identity", {}).get("compiler_options")):
                raise ValueError("saved B4 JVP graph does not match the B4 mathematical/compiler identity")
        compiler_options = identities["candidate"]["compiler_options"]
        if (b1_sidecar["identity"].get("compiler_options") != compiler_options
                or identity.get("saved_B1_mathematical_identity", {}).get("compiler_options") != compiler_options
                or b4_sidecar.get("device_kind") is None):
            raise ValueError("saved B1/B4 compiler options or device identity missing")
        paid_envelope = json.loads(args.paid_pair_envelope.read_text(encoding="utf-8"))
        paid_envelope_report = contract.paid_pair_envelope_report(
            {"B1": np.zeros(4), "B4": np.zeros(4)}, paid_envelope)
        if not args.b1_executable.is_file() or not args.b4_executable.is_file():
            raise FileNotFoundError("both exact saved full-state VJP executables are required")

        directions_host = contract.direction_matrix(dtype=np.float64)
        directions = jax.device_put(directions_host)
        if np.asarray(cases[contract.SAMPLE_INDEX][0]).dtype != directions_host.dtype:
            raise ValueError("FP64 tangent direction must match sample-3 FP64 state ABI")

        with ExitStack() as patches:
            objective = _candidate_objective(args, owner, transformer, context, head, patches)
            b1_fn, b4_fn = route_functions(jax, objective,
                retain_unscaled_b1=args.route_mode == "B1-completion")
            from jax.experimental.serialize_executable import serialize
            abi = pair.adjacent("shared_full_state_ad_execution")
            gates = pair.adjacent("probe_shared_union_micro4")
            device_parameters = jax.device_put(parameters)
            device_cases = tuple(jax.device_put(case) for case in cases)
            device_stacked = jax.device_put(stacked)
            arguments_b1 = (device_parameters, *device_cases[contract.SAMPLE_INDEX])
            arguments_b4 = (device_parameters, *device_stacked)
            route_fns = {"B1": b1_fn, "B4": b4_fn}
            route_args = {"B1": (*arguments_b1, directions[0]),
                          "B4": (*arguments_b4, directions[0])}
            jvp_execs = {}
            route_usage = {"B1": 0.0, "B4": 0.0}

            @contextmanager
            def route_stage(route, label, stage_cap=None):
                remaining = args.route_seconds - route_usage[route]
                if remaining <= 0:
                    raise TimeoutError(route + " cumulative route cap exhausted")
                limit = remaining if stage_cap is None else min(remaining, stage_cap)
                began = time.monotonic()
                with journal.deadline(limit, label):
                    yield
                elapsed = time.monotonic() - began
                route_usage[route] += elapsed
                journal.report["route_occupation_seconds"] = dict(route_usage)
                if route_usage[route] > args.route_seconds:
                    raise TimeoutError(route + " cumulative route cap exceeded")

            compile_routes = (("B4",) if args.route_mode == "B4-only" else
                              (("B1",) if args.route_mode == "B1-completion" else ("B1", "B4")))
            for route in compile_routes:
                attempt_path = args.output.with_suffix("." + route.lower() + ".jvp.compile-attempt.json")
                if attempt_path.exists():
                    raise FileExistsError(attempt_path)
                attempt_started = time.time()
                attempt = {"route": route, "attempt": 1, "mathematical_identity": identity,
                    "compile_cap": 1, "optimizer_update_cap": 0, "started_epoch": attempt_started,
                    "phase_markers": {"lower_started": False, "lower_finished": False,
                                       "backend_started": False, "backend_finished": False,
                                       "compiled": False},
                    "status": "attempt_started"}
                def persist_attempt():
                    owner.atomic_json(attempt_path, attempt)
                lowered_at = time.monotonic()
                attempt["phase_markers"]["lower_started"] = True
                attempt["lower_started_epoch"] = time.time()
                persist_attempt()
                with route_stage(route, "lower_" + route + "_JVP", args.lower_seconds):
                    lowered = jax.jit(route_fns[route],
                        compiler_options=compiler_options).lower(*route_args[route])
                lower_elapsed = time.monotonic() - lowered_at
                attempt["phase_markers"]["lower_finished"] = True
                attempt["lower_finished_epoch"] = time.time()
                attempt["lower_seconds"] = lower_elapsed
                persist_attempt()
                compile_at = time.monotonic()
                attempt["phase_markers"]["backend_started"] = True
                attempt["backend_started_epoch"] = time.time()
                persist_attempt()
                journal.report["backend_compile_attempted"] = True
                with route_stage(route, "compile_" + route + "_JVP", args.backend_seconds):
                    executable = lowered.compile()
                compile_elapsed = time.monotonic() - compile_at
                journal.report["model_compiles"][route + "_JVP"] = 1
                attempt["phase_markers"]["backend_finished"] = True
                attempt["backend_finished_epoch"] = time.time()
                attempt["phase_markers"]["compiled"] = True
                attempt.update(completed_epoch=time.time(), compile_seconds=compile_elapsed, status="compiled")
                persist_attempt()
                journal.report["route_measurement_status"][route + "_JVP"] = "compiled_unmeasured"
                with route_stage(route, "preserve_" + route + "_JVP"):
                    serialized = serialize(executable)
                    graph_path = args.output.with_suffix("." + route.lower() + ".jvp.executable.pkl")
                    _save(owner, graph_path, serialized)
                    owner.atomic_json(graph_path.with_suffix(graph_path.suffix + ".identity.json"), {
                        "identity": identity, "route": route, "compiler_options": compiler_options})
                journal.report.setdefault("compiled_routes", []).append({"route": route,
                    "executable": str(graph_path), "lower_seconds": lower_elapsed,
                    "backend_seconds": compile_elapsed,
                    "compiler_options": compiler_options})
                jvp_execs[route] = executable

            from jax.experimental.serialize_executable import deserialize_and_load
            saved_execs = {}
            load_times = {}
            began = time.monotonic()
            with journal.deadline(args.load_seconds, "load_saved_B1_VJP"), source.forbid_model_compile():
                serialized_b1 = pair.read_pickle(args.b1_executable)
                saved_execs["B1"] = deserialize_and_load(*serialized_b1)
                abi.validate_full_state_abi(jax, serialized_b1, saved_execs["B1"], arguments_b1, component="T7")
            load_times["B1"] = time.monotonic() - began
            began = time.monotonic()
            with journal.deadline(args.load_seconds, "load_saved_B4_VJP"), source.forbid_model_compile():
                serialized_b4 = pair.read_pickle(args.b4_executable)
                saved_execs["B4"] = deserialize_and_load(*serialized_b4)
                batch_ad.validate_batch_abi(jax, serialized_b4, saved_execs["B4"], arguments_b4)
            load_times["B4"] = time.monotonic() - began
            if args.route_mode == "B1-completion":
                began = time.monotonic()
                with journal.deadline(args.load_seconds, "load_saved_B4_JVP"), source.forbid_model_compile():
                    serialized_b4_jvp = pair.read_pickle(args.b4_jvp_executable)
                    jvp_execs["B4"] = deserialize_and_load(*serialized_b4_jvp)
                    b4_jvp_abi = validate_serialized_call_abi(
                        jax, serialized_b4_jvp, jvp_execs["B4"], route_args["B4"], "B4 JVP")
                load_times["B4_JVP"] = time.monotonic() - began
                journal.report["saved_B4_JVP_ABI"] = b4_jvp_abi
            journal.report["saved_executable_load_seconds"] = load_times

            memory_snapshots = []

            def memory_snapshot(label):
                stats = devices[0].memory_stats()
                peak = stats.get("peak_bytes_in_use") if isinstance(stats, dict) else None
                row = {"label": label, "stats": stats, "observed_peak_bytes": peak,
                       "under_24_GiB": isinstance(peak, (int, float)) and 0 <= peak < 24 * 2**30}
                memory_snapshots.append(row)
                journal.report["JAX_memory_snapshots"] = memory_snapshots
                if not row["under_24_GiB"]:
                    raise MemoryError("observed JAX peak is unavailable or not below24GiB")
                return row

            def saved_vjp(route, phase):
                arguments = arguments_b1 if route == "B1" else arguments_b4
                with route_stage(route, route + "_saved_VJP_" + phase):
                    with journal.deadline(args.route_seconds, route + "_saved_VJP_call_" + phase), source.forbid_model_compile():
                        raw = saved_execs[route](*arguments)
                        jax.block_until_ready(raw)
                    host = jax.device_get(raw)
                    path = args.output.with_suffix("." + route.lower() + ".vjp." + phase + ".result.pkl")
                    _save(owner, path, {"route": route, "phase": phase, "result": host})
                    journal.report["saved_VJP_calls"] += 1
                    journal.report["real_model_evaluated"] = True
                    if route == "B1":
                        pair.validate_result(jax, host, parameters, "candidate", cases[3][0])
                        projection = contract.project_b1_state_vjp(host[1][1])
                    else:
                        batch_ad.validate_batch_result(jax, host, parameters, stacked[0], gates)
                        projection = contract.project_b4_state_vjp(host[1][1])
                    memory_snapshot(route + "_saved_VJP_" + phase)
                    return {"primal_aux": host[0], "projection": projection, "artifact": str(path)}

            jax.block_until_ready((directions, arguments_b1, arguments_b4))
            vjp = {"begin": {route: saved_vjp(route, "begin") for route in ("B1", "B4")}}
            cycle_results = {"B1": [], "B4": []}
            strict_failures = []
            for cycle in range(1, contract.JVP_CYCLES + 1):
                order = ("B4",) if args.route_mode == "B4-only" else (("B1", "B4") if cycle % 2 else ("B4", "B1"))
                for route in order:
                    with route_stage(route, route + "_JVP_cycle" + str(cycle)):
                        direction_rows = []
                        raw_direction_rows = []
                        primal_rows = []
                        for direction_index, coordinate in enumerate(contract.STATE_COORDINATES):
                            arguments = arguments_b1 if route == "B1" else arguments_b4
                            call_started = time.monotonic()
                            with journal.deadline(args.route_seconds, route + "_JVP_call_" + str(cycle)), source.forbid_model_compile():
                                result = jvp_execs[route](*arguments, directions[direction_index])
                                jax.block_until_ready(result)
                            host_result = jax.device_get(result)
                            primal_rows.append(host_result[0])
                            if route == "B1" and args.route_mode == "B1-completion":
                                raw_tangent, scaled_tangent = host_result[1]
                                raw_tangent = np.asarray(raw_tangent)
                                tangent = np.asarray(scaled_tangent)
                            else:
                                tangent = np.asarray(host_result[1])
                                raw_tangent = tangent
                            direction_rows.append(tangent)
                            raw_direction_rows.append(raw_tangent)
                            journal.report.setdefault("JVP_call_timings", []).append({"route": route,
                                "cycle": cycle, "coordinate": coordinate,
                                "seconds": time.monotonic() - call_started})
                            journal.report["JVP_calls"] += 1
                            journal.report["real_model_evaluated"] = True
                            journal.report["route_measurement_status"][route + "_JVP"] = "measured"
                        # Keep every primal/auxiliary and tangent before any bridge or
                        # strict parity gate can stop the diagnostic.
                        cycle_path = args.output.with_suffix("." + route.lower() + ".jvp.cycle" + str(cycle) + ".result.pkl")
                        _save(owner, cycle_path, {"route": route, "cycle": cycle,
                            "coordinates": list(contract.STATE_COORDINATES),
                            "primal_and_auxiliary": primal_rows, "tangents": direction_rows,
                            "raw_tangents": raw_direction_rows,
                            "tangent_scale": "raw_and_div4" if route == "B1" and args.route_mode == "B1-completion"
                                else "mean_objective_scale"})
                        memory_snapshot(route + "_JVP_cycle" + str(cycle))
                    if any(value.shape != () or value.dtype != np.dtype("float64") or not np.isfinite(value)
                           for value in direction_rows):
                        raise ValueError(route + " JVP tangent scalar ABI/finite gate failed")
                    direction_rows = [float(value) for value in direction_rows]
                    row = {"route": route, "cycle": cycle, "artifact": str(cycle_path),
                           "tangents": direction_rows,
                           "raw_tangents": [float(value) for value in raw_direction_rows],
                           "tangent_scale": "B1_raw_and_div4" if route == "B1" and args.route_mode == "B1-completion"
                               else "B4_mean_objective",
                           "primal_bridges": []}
                    for coordinate_index, coordinate in enumerate(contract.STATE_COORDINATES):
                        bridge_routes = primal_bridge_routes(route)
                        endpoint_routes = (("B1", "B4") if args.route_mode in ("B4-only", "B1-completion")
                                           else (route,))
                        bridges = {phase + "_" + bridge_route: compare_primal_bridge(jax, route,
                            primal_rows[coordinate_index], vjp[phase][bridge_route]["primal_aux"])
                            for phase in ("begin",) for bridge_route in bridge_routes}
                        # End baselines are filled after all direction calls.
                        tangent_value = direction_rows[coordinate_index]
                        endpoint_checks = {bridge_route: contract.compare_scalar(tangent_value,
                            vjp["begin"][bridge_route]["projection"][coordinate_index], coordinate=coordinate)
                            for bridge_route in endpoint_routes}
                        row["primal_bridges"].append({"coordinate": coordinate, "comparisons": bridges,
                            "JVP_vs_begin_saved_VJP": endpoint_checks})
                        if (not all(value["passed"] for value in bridges.values())
                                or not all(value["passed"] for value in endpoint_checks.values())):
                            strict_failures.append({"route": route, "cycle": cycle, "coordinate": coordinate,
                                "kind": "begin_bridge_or_projection"})
                    first = cycle_results[route][0] if cycle_results[route] else row
                    row["JVP_vs_first_cycle"] = []
                    for index, coordinate in enumerate(contract.STATE_COORDINATES):
                        repeat = contract.compare_scalar(direction_rows[index],
                            first["tangents"][index], coordinate=coordinate)
                        row["JVP_vs_first_cycle"].append(repeat)
                        if not repeat["passed"]:
                            strict_failures.append({"route": route, "cycle": cycle,
                                "coordinate": coordinate, "kind": "same_route_JVP_repeat"})
                    cycle_results[route].append(row)
                    journal.report.setdefault("cycle_results", []).append(row)

            vjp["end"] = {route: saved_vjp(route, "end") for route in ("B1", "B4")}
            checked_routes = ("B1", "B4")
            for route in checked_routes:
                for row in cycle_results[route]:
                    raw_cycle = pair.read_pickle(Path(row["artifact"]))
                    for index, coordinate in enumerate(contract.STATE_COORDINATES):
                        primal = raw_cycle["primal_and_auxiliary"][index]
                        bridge_routes = primal_bridge_routes(route)
                        endpoint_routes = (("B1", "B4") if args.route_mode in ("B4-only", "B1-completion")
                                           else (route,))
                        bridges = {"end_" + bridge_route: compare_primal_bridge(jax, route, primal,
                            vjp["end"][bridge_route]["primal_aux"]) for bridge_route in bridge_routes}
                        endpoint_checks = {bridge_route: contract.compare_scalar(row["tangents"][index],
                            vjp["end"][bridge_route]["projection"][index], coordinate=coordinate)
                            for bridge_route in endpoint_routes}
                        row["primal_bridges"][index]["comparisons"].update(bridges)
                        row["primal_bridges"][index]["JVP_vs_end_saved_VJP"] = endpoint_checks
                        if (not all(value["passed"] for value in bridges.values())
                                or not all(value["passed"] for value in endpoint_checks.values())):
                            strict_failures.append({"route": route, "cycle": row["cycle"],
                                "coordinate": coordinate, "kind": "end_bridge_or_projection"})
                for cycle_row in cycle_results[route]:
                    env = contract.paid_pair_envelope_report(
                        {"B1": np.asarray(cycle_row["tangents"], dtype=np.float64),
                         "B4": np.asarray(cycle_row["tangents"], dtype=np.float64)}, paid_envelope)
                    journal.report.setdefault("paid_envelope_membership", []).append({
                        "route": route, "cycle": cycle_row["cycle"],
                        "route_envelopes": env["routes"][route]})

            cross_route = []
            for cycle in range(contract.JVP_CYCLES) if args.route_mode in ("both", "B1-completion") else ():
                for index, coordinate in enumerate(contract.STATE_COORDINATES):
                    comparison = contract.compare_scalar(
                        cycle_results["B1"][cycle]["tangents"][index],
                        cycle_results["B4"][cycle]["tangents"][index], coordinate=coordinate)
                    cross_route.append({"cycle": cycle + 1, "coordinate": coordinate, "comparison": comparison})
                    if not comparison["passed"]:
                        strict_failures.append({"cycle": cycle + 1, "coordinate": coordinate, "kind": "cross_route_JVP"})
            endpoint_route_checks = {}
            for phase in ("begin", "end"):
                comparison = contract.compare_projections(vjp[phase]["B1"]["projection"], vjp[phase]["B4"]["projection"])
                endpoint_route_checks[phase] = comparison
                if not comparison["passed"]:
                    strict_failures.append({"phase": phase, "kind": "cross_route_saved_VJP"})
            endpoint_self_checks = {}
            for route in ("B1", "B4"):
                comparison = contract.compare_projections(vjp["begin"][route]["projection"],
                                                         vjp["end"][route]["projection"])
                endpoint_self_checks[route] = comparison
                if not comparison["passed"]:
                    strict_failures.append({"route": route, "kind": "saved_VJP_self_drift"})
            journal.report["cross_route_same_cycle_JVP"] = cross_route
            journal.report["cross_route_saved_VJP_endpoints"] = endpoint_route_checks
            journal.report["saved_VJP_self_drift"] = endpoint_self_checks
            if args.route_mode == "B4-only":
                journal.report["B1_JVP"] = {"status": "unmeasured",
                    "reason": "B1 backend compilation timed out in the preserved original attempt; retry forbidden"}
            journal.report["paid_pair_envelope_source"] = {"path": str(args.paid_pair_envelope),
                "sha256": pair.sha(args.paid_pair_envelope), "descriptive_only": True}
            journal.report["strict_failures"] = strict_failures
            success_status = "partial_diagnostic_complete" if args.route_mode == "B4-only" else "diagnostic_complete"
            journal.report.update(status="numerical_failed" if strict_failures
                else success_status, GPU_executed=True,
                real_model_evaluated=True,
                JVP_calls=calls, saved_VJP_calls=4,
                inference="diagnostic_only_no_root_cause_or_scientific_harm_claim",
                route_conclusion="B1_JVP_unmeasured_no_complete_B1_B4_JVP_conclusion"
                    if args.route_mode == "B4-only" else "both_routes_measured")
            journal.emit("diagnostic_results_preserved")
            return journal.report
    except Exception as error:
        journal.report.update(status="error", error_type=type(error).__name__, error=str(error))
        journal.emit("error")
        raise
    finally:
        journal.timer.cancel()


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        result = run(arguments)
    except TimeoutError as error:
        receipt = {"status": "timeout", "error": str(error), "output": str(arguments.output)}
        print(json.dumps(receipt, sort_keys=True), file=sys.stderr)
        return 124
    except Exception as error:
        receipt = {"status": "error", "error_type": type(error).__name__,
                   "error": str(error), "output": str(arguments.output)}
        print(json.dumps(receipt, sort_keys=True), file=sys.stderr)
        return 1
    receipt = {key: result.get(key) for key in (
        "status", "GPU_executed", "GPU_occupation_started", "gpu_backend_started",
        "backend_compile_attempted", "real_model_evaluated",
        "route_mode", "route_measurement_status", "model_compiles", "JVP_calls", "saved_VJP_calls",
        "planned_model_compiles", "planned_JVP_calls", "planned_saved_VJP_calls",
        "saved_executable_load_count", "saved_executables_to_load", "occupation_caps",
        "saved_B4_JVP_reused_without_recompile", "optimizer_updates", "training_updates")}
    receipt["output"] = str(arguments.output)
    print(json.dumps(receipt, sort_keys=True))
    if result.get("status") == "numerical_failed":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
