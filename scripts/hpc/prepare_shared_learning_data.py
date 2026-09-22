"""Read the fixed100 existing TRAIN artifacts on CPU; no model execution."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.shared_learning_plan import load_learning_configuration
from scripts.hpc.shared_transformer_runtime import _load_prepared_artifacts


def day_keys(records, *, horizon=1):
    return {(str(record["landpoint_id"]), int(record["year"]), int(start) + offset)
            for record in records for start in record["start_rows"] for offset in range(horizon)}


def _signature(tree):
    import jax

    leaves, definition = jax.tree_util.tree_flatten(tree)
    return str(definition), tuple((tuple(np.shape(value)), str(np.asarray(value).dtype)) for value in leaves)


def artifact_signatures(artifact):
    """Shapes of all consumed arrays, accounting for distinct retained axes."""
    import jax

    daily = jax.tree_util.tree_map(lambda value: np.asarray(value)[0], artifact.batch.one_day)
    signatures = {"daily": _signature(daily)}
    for name in ("seven_day", "thirty_day"):
        batch = getattr(artifact.batch, name)
        # Other window members own a window/day prefix; retained owns only window.
        # Removing the window axis preserves their relevant remaining shape.
        signatures[name] = _signature(jax.tree_util.tree_map(lambda value: np.asarray(value)[0], batch))
    return signatures


def run(args):
    began = time.perf_counter()
    evidence = load_learning_configuration(args.protocol, args.joint_admission_report,
                                          args.prepared_manifest, args.validation_manifest)
    config, plan = evidence["config"], evidence["plan"]
    validation_keys = {(row["landpoint_id"], int(row["year"]))
                       for row in evidence["validation_panel"]["rows"]}
    ids = plan["training_artifact_ids"]
    reference_id = int(evidence["admission"]["artifact_ids"][0])
    references, _ = _load_prepared_artifacts(args.prepared_manifest, [reference_id])
    reference_signatures = artifact_signatures(references[0])
    classes = {key: [value] for key, value in reference_signatures.items()}
    rows = []
    consumed = {key: set() for key in ("daily", "seven_day", "thirty_day")}
    report = {"status": "running", "model_executed": False, "numerical_gradients_checked": False,
              "scientific_acceptance": False, "config": config, "plan": plan,
              "validation_panel": evidence["validation_panel"], "admitted_reference_artifact": reference_id}
    try:
        for ordinal, identifier in enumerate(ids, start=1):
            artifacts, _ = _load_prepared_artifacts(args.prepared_manifest, [identifier])
            artifact = artifacts[0]
            batch_count = int(np.shape(artifact.batch.one_day.inputs.continuous_state)[0])
            if batch_count != config["daily_rows_per_update"]:
                raise ValueError(f"artifact{identifier} has{batch_count} daily rows, expected32")
            components = artifact.schedule_entry["components"]
            daily_records = components["one_day"]
            if sum(len(record["start_rows"]) for record in daily_records) != batch_count:
                raise ValueError("daily array and schedule populations differ")
            keys = {"daily": day_keys(daily_records)}
            for component, horizon in (("seven_day", 7), ("thirty_day", 30)):
                record = components[component][0]
                keys[component] = day_keys([{**record, "start_rows": [record["start_rows"][0]]}], horizon=horizon)
                day_index = np.asarray(getattr(artifact.batch, component).inputs.day_index)[0].reshape(-1)
                if day_index.size != horizon or not np.all(np.diff(day_index) == 1):
                    raise ValueError("selected recurrent training window is not consecutive")
            point_years = {(point, year) for current in keys.values() for point, year, _ in current}
            if point_years & validation_keys:
                raise ValueError("training sample intersects held-out development point-years")
            shape_classes = {}
            for component, signature in artifact_signatures(artifact).items():
                if signature not in classes[component]:
                    classes[component].append(signature)
                shape_classes[component] = classes[component].index(signature)
            for key, current in keys.items():
                consumed[key].update(current)
            rows.append({"artifact_id": identifier, "ordinal": ordinal, "daily_rows": batch_count,
                         "shape_classes": shape_classes, "point_years": sorted(point_years)})
            if ordinal % 10 == 0:
                report.update(checked_artifacts=ordinal, elapsed_seconds=time.perf_counter() - began)
                _atomic_json(args.output, report)
                print(f"shared_learning_data checked={ordinal}/100 seconds={report['elapsed_seconds']:.1f}", flush=True)
        union = set().union(*consumed.values())
        report.update(
            status="passed", checked_artifacts=len(rows), artifacts=rows,
            consumed_unique_days={key: len(value) for key, value in consumed.items()},
            all_components_unique_days=len(union),
            unique_landpoints=len({point for point, _, _ in union}),
            unique_point_years=len({(point, year) for point, year, _ in union}),
            years=sorted({year for _, year, _ in union}),
            shape_class_counts={key: len(value) for key, value in classes.items()},
            all_match_admitted_shapes=all(len(value) == 1 for value in classes.values()),
            held_out_point_year_overlap=0, elapsed_seconds=time.perf_counter() - began,
        )
        _atomic_json(args.output, report)
        print(f"shared_learning_data completed seconds={report['elapsed_seconds']:.1f}", flush=True)
        return report
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.perf_counter() - began)
        _atomic_json(args.output, report)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "joint-admission-report", "prepared-manifest", "validation-manifest", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
