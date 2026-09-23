"""Metadata-only plans for bounded shared-Transformer learning runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build_learning_plan(prepared_manifest, configuration):
    records = prepared_manifest["entries"]
    identifiers = sorted(int(record["update"]) for record in records)
    count = int(configuration["max_updates"])
    if len(identifiers) != len(set(identifiers)) or len(identifiers) < count:
        raise ValueError("learning requires distinct available TRAIN artifact identities")
    rng = np.random.Generator(np.random.PCG64(int(configuration["sampling_seed"])))
    chosen = rng.permutation(identifiers)[:count]
    selected = [int(identifier) for identifier in chosen]
    by_id = {int(record["update"]): record for record in records}
    years = sorted({int(by_id[identifier]["runtime_year"]) for identifier in selected})
    return {
        "training_artifact_ids": selected,
        "fit_artifact_ids": selected[:int(configuration["fit_artifact_count"])],
        "prepared_manifest_identity": prepared_manifest["identity"],
        "summary": {
            "available_artifacts": len(identifiers),
            "selected_artifacts": count,
            "runtime_years": years,
            "daily_draws": count * int(configuration["daily_rows_per_update"]),
            "windows_per_horizon": count,
            "unique_daily_rows": None,
            "unique_day_coverage_requires_array_schedule_inspection": True,
            "sampling_seed": int(configuration["sampling_seed"]),
            "is_epoch": False,
        },
    }


def load_learning_configuration(protocol_path, joint_admission_report_path,
                                prepared_manifest_path, validation_manifest_path):
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    path = Path(protocol_path)
    config = read(path)
    schema = config.get("schema_version")
    if schema == "shared_learning_v1":
        schedule = (100, 5, [0, 20, 100])
    elif schema == "shared_learning_v2":
        schedule = (500, 25, [0, 100, 250, 500])
        if config.get("daily_microbatch_size") != 4:
            raise ValueError("shared learning v2 requires four-row daily microbatches")
        if (config.get("model_protocol") != "shared_transformer_split_width_v1.json"
                or config.get("joint_protocol") != "shared_split_recurrent_admission_v1.json"
                or config.get("initialization")
                != "fresh_model_seed_and_fresh_AdamW_no_exploratory_weights"
                or config.get("gradient_accumulation")
                != "mean_32_daily_gradients_evaluated_in_micro4_plus_T7_plus_T30_then_one_AdamW_update"):
            raise ValueError("shared learning v2 method identity differs from the declared split method")
    else:
        raise ValueError("unsupported shared learning schema")
    if (config["max_updates"] != schedule[0]
            or config["daily_rows_per_update"] != 32
            or config["checkpoint_interval"] != schedule[1]
            or config["milestones"] != schedule[2]
            or config["recurrent_windows_per_horizon"] != 1
            or config["recurrent_window_index"] != 0 or config["fit_artifact_count"] != 8
            or config["paid_submission"] or config["teacher_regeneration"]
            or config["sealed_test_access"] or not config["all_parameters_trainable"]):
        raise ValueError("configuration differs from the bounded shared learning method")
    daily = read(path.parent / config["model_protocol"])
    joint = read(path.parent / config["joint_protocol"])
    admission = read(joint_admission_report_path)
    if (admission.get("status") != "passed"
            or admission.get("classification") != "shared_daily_T7_T30_two_update_technical_admission_only"
            or admission.get("completed_updates") != 2
            or admission.get("protocol") != joint or admission.get("daily_protocol") != daily
            or admission.get("checkpoint_inherited") is not False
            or admission.get("trainable_parameter_count") != daily["architecture"]["parameter_count"]):
        raise ValueError("shared joint numerical admission must be passed for this method")
    manifest = read(prepared_manifest_path)
    if manifest["identity"] != admission["evidence_identity"]["prepared_manifest_identity"]:
        raise ValueError("learning TRAIN pool differs from the admitted dataset")
    from scripts.hpc.fresh_method_validation_pool import FreshMethodValidationPool

    pool = FreshMethodValidationPool(validation_manifest_path)
    if len(pool.rows) != 48 or int(pool.panel["windows_per_slice_per_horizon"]) != 8:
        raise ValueError("bounded learning requires the complete existing48-window panel")
    return {
        "config": config,
        "admission": admission,
        "daily_protocol": daily,
        "plan": build_learning_plan(manifest, config),
        "validation_panel": {
            "panel_sha256": pool.panel["panel_sha256"],
            "rows": pool.rows,
            "role": "held_out_development",
            "used_for_parameter_updates": False,
            "sealed_test_used": False,
        },
    }
