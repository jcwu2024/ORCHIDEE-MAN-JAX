"""CPU-only actual-data preparation for the shared daily/T7/T30 admission.

Reuse accepted TRAIN calibration; no weights, model executions, new Teacher
arrays, GPU compilation or optimizer updates are produced by this command.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.hpc.run_shared_transformer_admission import ROOT, shared_config
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.shared_transformer_runtime import load_shared_runtime, select_artifact_ids


def load_protocol_and_calibration(protocol_path, calibration_path, prepared_manifest):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["horizons"] != [7, 30] or protocol["window_indices"] != [0]:
        raise ValueError("bounded recurrent admission requires T7/T30 and window0")
    if (protocol["joint_daily_coefficients"] != [1/3, 1/9, 1/9]
            or protocol["joint_recurrent_flux_stock_coefficients"] != [1/9, 1/9]):
        raise ValueError("recurrent protocol must preserve equal three-group/horizon weights")
    daily = json.loads((protocol_path.parent / protocol["daily_protocol"]).read_text(encoding="utf-8"))
    saved = json.loads(calibration_path.read_text(encoding="utf-8"))
    if saved["status"] != "passed" or saved["protocol"] != daily:
        raise ValueError("calibration requires a passed unchanged daily-v3 protocol")
    if saved["calibration"]["split"] != "train/train":
        raise ValueError("recurrent scales must be TRAIN-only")
    manifest = json.loads(prepared_manifest.read_text(encoding="utf-8"))
    ids = select_artifact_ids(manifest["entries"], daily["calibration"]["artifact_count"])
    if tuple(saved["calibration"]["artifact_ids"]) != ids:
        raise ValueError("saved calibration artifact selection differs")
    if saved["data"]["prepared_manifest_identity"] != manifest["identity"]:
        raise ValueError("saved calibration belongs to a different prepared dataset")
    return protocol, daily, saved, ids[:2]


def selected_window_metadata(artifact, component, horizon):
    record = artifact.schedule_entry["components"][component][0]
    return {"artifact_id": int(artifact.schedule_entry["update"]),
            "component": component, "horizon": horizon, "window_index": 0,
            "landpoint_id": record["landpoint_id"], "year": int(record["year"]),
            "start_row": int(record["start_rows"][0])}


def run(args):
    started = time.perf_counter()
    report = {"status": "running", "classification": "CPU_preparation_only",
              "completed_updates": 0, "model_forward_executed": False,
              "numerical_gradient_checked": False, "scientific_acceptance": False,
              "source_git_head": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}

    def emit(stage, **values):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **values)
        _atomic_json(args.output, report)
        print(f"shared_recurrent_prepare stage={stage} seconds={report['elapsed_seconds']:.1f}", flush=True)

    try:
        protocol, daily, saved, ids = load_protocol_and_calibration(
            args.protocol, args.calibration_report, args.prepared_manifest)
        emit("load_existing_train_artifacts", protocol=protocol, daily_protocol=daily,
             calibration_source=str(args.calibration_report), calibration_reused=True,
             scales=saved["scales"], artifact_ids=ids)
        args.shape_only = True
        runtime, parent, template, artifacts, metadata = load_shared_runtime(args, ids)
        jax = runtime["jax"]
        from research.daily_coarse_graining.daily_operator.shared_persistent_outputs import (
            build_shared_persistent_coordinates,
        )
        from research.daily_coarse_graining.daily_operator.shared_process_head import build_shared_process_head
        from research.daily_coarse_graining.daily_operator.shared_recurrent_training import (
            make_shared_joint_daily_objective,
            make_shared_recurrent_objective,
            prepare_shared_window,
        )
        from research.daily_coarse_graining.daily_operator.shared_training import derive_shared_state_change_scales

        with jax.default_device(jax.devices("cpu")[0]):
            coordinates = build_shared_persistent_coordinates(
                parent.learned_schema, parent.contract, runtime["base_statistics"].arrays["state_delta"])
            head = build_shared_process_head(
                parent, template, config=shared_config(daily["architecture"]),
                thermal_transform=daily["architecture"]["thermal_transform"],
                persistent_coordinates=coordinates, source_bare_cimean=True)
            state_scales = derive_shared_state_change_scales(runtime, head)
            shapes = jax.eval_shape(lambda: head.initialize(runtime["probe_assembled"], seed=daily["seed"]))
            count = sum(int(np.prod(value.shape)) for value in jax.tree_util.tree_leaves(shapes))
            if count != daily["architecture"]["parameter_count"]:
                raise ValueError("shared recurrent parameter tree differs from daily protocol")
            emit("shared_parameter_shapes", parameter_count=count, data=metadata)
            daily_objective = make_shared_joint_daily_objective(
                runtime, head, template, state_scales, saved["scales"])
            _rollout, recurrent = make_shared_recurrent_objective(head, template, state_scales, saved["scales"])
            sample = jax.tree_util.tree_map(lambda value: value[0], artifacts[0].batch.one_day)
            jax.eval_shape(jax.value_and_grad(daily_objective, has_aux=True), shapes, sample)
            rows = []
            for artifact in artifacts:
                for horizon, component in ((7, "seven_day"), (30, "thirty_day")):
                    row = selected_window_metadata(artifact, component, horizon)
                    case = prepare_shared_window(
                        head, getattr(artifact.batch, component), window_index=0, horizon=horizon)
                    arguments = (shapes, *case)
                    result_shape = jax.eval_shape(recurrent, *arguments)
                    # One gradient trace per horizon suffices for equal-shaped second data.
                    if artifact is artifacts[0]:
                        jax.eval_shape(jax.value_and_grad(recurrent, has_aux=True), *arguments)
                    row.update(target_state_shape=list(case.targets["state"].shape),
                               loss_dtype=str(result_shape[0].dtype),
                               hard_counter_names=sorted(result_shape[1]["hard"]))
                    rows.append(row)
                    emit("window_shapes_completed", windows=rows)
        emit("completed", status="passed", next_gate="wait_for_daily_AD_then_T7_T30_GPU_admission")
        return report
    except Exception as error:
        emit("exception", status="error", error_type=type(error).__name__, error=str(error))
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-manifest", "acceptance-report", "base-statistics", "process-statistics",
                 "fast-statistics", "prepared-manifest", "calibration-report", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--protocol", type=Path,
                        default=ROOT / "manifests/coarse_graining/shared_recurrent_admission_v1.json")
    return result


if __name__ == "__main__":
    outcome = run(parser().parse_args())
    raise SystemExit(0 if outcome["status"] == "passed" else 1)
