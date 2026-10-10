"""CPU-only actual T7 first-two-day canonical feedback and biomass capture.

Every day recomputes its neural heads from the previous predicted state and
the declared T7 forcing/template. This is forward-only bounded evidence,
without Teacher state injection, neural-output replay, AD or training.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

if str(Path(__file__).parent) not in sys.path: sys.path.insert(0, str(Path(__file__).parent))
import probe_shared_biomass_state_amplification as capture


def run(args):
    capture.STARTED = time.monotonic()
    if not 10 <= args.wall_seconds <= 300: raise ValueError("T7 prefix wall must be bounded10..300s")
    if args.neural_execution != "component_jit": raise ValueError("bounded actual feedback requires component_jit")
    if args.neural_cache or args.physical_state_vjp or args.capture_turnover_cotangent or args.additional_capture or args.reference_state_diagnostics:
        raise ValueError("T7 prefix uses fresh daily neural execution; daily/VJP/replay modes unavailable")
    if bool(args.turnover_overlay) != bool(args.turnover_base_sha256): raise ValueError("overlay baseline hash required")
    if args.output.exists() or args.output.with_suffix(".npz").exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    import jax
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "cpu": raise ValueError("CPU backend required")
    source = capture.adjacent("probe_shared_original_full_state_schedule")
    capture.adjacent("shared_full_state_ad_execution")
    capture.adjacent("training_artifact_io")
    owner = capture.adjacent("probe_shared_expert_real_daily")
    manifest = source.read_manifest(args.manifest)
    if str(args.chain_snapshot) != manifest["resume"][args.schedule]: raise ValueError("exact declared snapshot required")
    profile, reference = source.read_reference(manifest)
    source.validate_resume_metadata(manifest, reference)
    source.validate_original_assets(manifest, reference)
    root = Path(os.environ.get("ORCHIDEE_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != reference["source_git_head"]: raise ValueError("source checkout provenance differs")
    report = {"schema_version": "shared_biomass_actual_T7_prefix2_feedback_v1", "status": "running",
        "scope": "actual_T7_first_two_canonical_forward_days_not_full_horizon_acceptance",
        "artifact_id": args.artifact_id, "schedule": args.schedule, "source_git_head": head,
        "chain_snapshot": str(args.chain_snapshot), "prepared_context": str(args.prepared_context),
        "GPU_executed": False, "network_AD": False, "full_network_gradient_compilations": 0,
        "neural_output_replay": False, "Teacher_state_injection": False,
        "precision": "unchanged_FP32_neural_FP64_physical", "scientific_acceptance": False,
        "adoption": False, "historical_binary_cause_established": False, "stage_events": [], "days": []}
    collector = capture.BoundaryCollector(on_boundary=lambda: capture.persist_capture(args.output, report, collector))
    def emit(stage, **metadata):
        report["stage"] = stage
        report["stage_events"].append({"stage": stage, "rollout_day": report.get("current_day"),
            "elapsed_seconds": time.monotonic()-capture.STARTED, **metadata})
        capture.persist_capture(args.output, report, collector)
    def timeout():
        report["status"] = "incomplete_timeout"
        try: capture.persist_capture(args.output, report, collector)
        finally: os._exit(124)
    timer = threading.Timer(max(.01, args.wall_seconds-(time.monotonic()-capture.STARTED)), timeout)
    timer.daemon = True
    timer.start()
    try:
        emit("load_actual_T7_inputs_and_chain")
        context = capture.read_pickle(args.prepared_context)
        expected = owner.context_identity(reference, Path(manifest["original_checkpoint"]), reference["method"]["training_plan"]["training_artifact_ids"][0])
        if context.get("identity") != expected or Path(context["daily_executable"]).name != Path(profile["executables"]["daily"]["executable"]).name:
            raise ValueError("prepared context provenance differs")
        bundle = capture.read_pickle(manifest["input_bundle"])
        source.validate_bundle(jax, bundle, manifest, reference)
        ids = list(bundle["artifact_ids"])
        if ids.count(args.artifact_id) != 1: raise ValueError("one actual artifact required")
        case = bundle["payloads"][ids.index(args.artifact_id)]["cases"][7]
        fingerprint = hashlib.sha256()
        for value in jax.tree_util.tree_leaves(case):
            array = np.ascontiguousarray(value)
            fingerprint.update(str((array.shape, array.dtype.str)).encode())
            fingerprint.update(array.tobytes())
        report["input_binding"] = {"artifact_id": args.artifact_id, "artifact_index": ids.index(args.artifact_id),
            "horizon": 7, "executed_prefix_days": 2, "actual_case_sha256": fingerprint.hexdigest(),
            "source": "existing actual prepared T7 five-element case"}
        parameters, version = capture.validate_snapshot(jax, capture.read_pickle(args.chain_snapshot), manifest, args.schedule)
        report["parameter_version"] = version
        report["biomass_layout"] = capture.biomass_layout(context["head"].contract)
        case, parameters = jax.device_put(case), jax.device_put(parameters)
        state, discrete = case.initial_continuous, case.initial_discrete
        collector.put("actual_T7_initial_state", state)
        report["actual_T7_templates_and_retained"] = capture.capture_tree(collector, "actual_T7_templates_and_retained", case.days)
        adapter = owner.load_staged_adapter(args.field_adapter)
        assembler = owner.load_backbone_assembler(args.backbone_adapter)
        admission = json.loads(Path(manifest["components"]["daily"]["report"]).read_text())
        report["overlay_source_hashes"] = {}
        for key, path in (("field_expert_adapter", args.field_adapter), ("backbone_adapter", args.backbone_adapter)):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            historic = admission["source_artifacts"][key]["sha256"]
            report["overlay_source_hashes"][key] = {"sha256": digest, "historic_sha256": historic,
                "matches_historic": digest == historic, "path": str(path)}
        def save_neural(name, value):
            prefix = f"day{report['current_day']}.{name}"
            report.setdefault("actual_neural_outputs_by_day", {})[prefix] = capture.capture_tree(collector, prefix, value)
            emit(name+"_captured")
        adapter = capture.timed_neural_component(adapter, "neural_decoder", "component_jit", emit,
            capture=lambda value: save_neural("raw_outputs", value))
        assembler = capture.timed_neural_component(assembler, "neural_backbone", "component_jit", emit,
            capture=lambda value: save_neural("context", value))
        from research.daily_coarse_graining.daily_operator import shared_process_head
        from research.daily_coarse_graining.daily_operator.canonical_rollout import make_canonical_step
        from research.daily_coarse_graining.daily_operator.recurrent_validation import physical_counts
        from research.daily_coarse_graining.daily_operator.shared_recurrent_training import _shared_stock_negative
        from research.daily_coarse_graining.daily_operator.shared_trajectory_loss import shared_component_trajectories
        from jax_orchidee.stomate import carbon_kernels
        with ExitStack() as patches:
            if args.turnover_overlay:
                report["diagnostic_turnover_overlay"] = patches.enter_context(capture.install_turnover_overlay(
                    carbon_kernels, args.turnover_overlay, args.turnover_base_sha256))
            step = make_canonical_step(context["head"], context["template"])
            patches.enter_context(jax.disable_jit())
            patches.enter_context(owner.candidate_process_head_patches(shared_process_head, adapter, assembler))
            patches.enter_context(capture.instrument_source_kernels(collector, carbon_kernels))
            observations, predicted = [], []
            for day in range(2):
                report["current_day"] = day
                collector.active_day = day
                began = time.monotonic()
                before = len(collector.records)
                current_day = jax.tree_util.tree_map(lambda value: value[day], case.days)
                collector.put(f"day{day}.actual_feedback_initial_state", state)
                emit("canonical_feedback_step_entered", day_index=int(np.asarray(current_day[0].day_index)))
                state, discrete, obs = step(parameters, state, discrete, current_day)
                jax.block_until_ready((state, discrete, obs))
                predicted.append(state)
                observations.append(obs)
                stacked = jax.tree_util.tree_map(lambda *values: jax.numpy.stack(values), *observations)
                stacked["state"] = jax.numpy.stack(predicted)
                templates = jax.tree_util.tree_map(lambda value: value[:day+1], case.days[0])
                expected_discrete = jax.tree_util.tree_map(lambda value: value[:day+1], case.loss_context.expected_discrete)
                targets = {"defined": case.targets["defined"][:day+1]}
                hard = physical_counts(stacked, targets, expected_discrete, templates, context["head"].contract, context["state_scales"])
                hard["shared_stock_negative"] = _shared_stock_negative(shared_component_trajectories(stacked, context["head"].contract))
                hard = {key: int(np.asarray(value)) for key, value in hard.items()}
                layout = report["biomass_layout"]
                source_pft = layout["selected_pft_indices"][layout["unravelled_coordinate"][layout["axis_names"].index("nvm")]]
                turnover_bindings = []
                for boundary in collector.records[before:]:
                    if boundary["symbol"] != "turnover_leaf_age_fall" or boundary["status"] != "returned": continue
                    actual_biomass = collector.arrays[boundary["outputs"]["0"]["array"]]
                    leaf_mass = float(actual_biomass[0,source_pft,carbon_kernels.ILEAF,carbon_kernels.ICARBON])
                    turnover_bindings.append({"call_index": boundary["index"], "source_pft_index": source_pft,
                        "actual_postturnover_leaf_mass": leaf_mass, "has_leaf_biomass": leaf_mass > 0.,
                        "existing_ratio_tangent_representable": abs(leaf_mass) >= np.sqrt(np.finfo(actual_biomass.dtype).tiny)})
                report["days"].append({"rollout_day": day, "day_index": int(np.asarray(current_day[0].day_index)),
                    "year": int(np.asarray(current_day[0].year)), "seconds": time.monotonic()-began,
                    "cold_or_warm": "cold" if day == 0 else "warm",
                    "actual_previous_prediction_feedback": day > 0,
                    "predicted_leaf_coordinate2243": float(np.asarray(state)[2243]),
                    "predicted": capture.capture_tree(collector, f"day{day}.predicted", state),
                    "observations": capture.capture_tree(collector, f"day{day}.observations", obs),
                    "hard_counts": hard, "boundary_start": before, "boundary_stop": len(collector.records)})
                report["days"][-1]["actual_turnover_bindings"] = turnover_bindings
                emit("canonical_feedback_step_returned")
                if any(hard.values()):
                    report["status"] = "forward_gate_failed"
                    capture.persist_capture(args.output, report, collector)
                    return report
        report["status"] = "captured"
        capture.persist_capture(args.output, report, collector)
        return report
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        capture.persist_capture(args.output, report, collector)
        raise
    finally: timer.cancel()


if __name__ == "__main__":
    parser = capture.parser()
    parser.description = __doc__
    arguments = parser.parse_args()
    result = run(arguments)
    raise SystemExit(0 if result["status"] == "captured" else 1)
