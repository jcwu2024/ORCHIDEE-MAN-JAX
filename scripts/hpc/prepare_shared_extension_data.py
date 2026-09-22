"""Audit a seeded extension against completed pilot data; no model execution."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.hpc.prepare_shared_learning_data import artifact_signatures, day_keys
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.shared_learning_plan import build_learning_plan
from scripts.hpc.shared_transformer_runtime import _load_prepared_artifacts


def summarize(consumed):
    union = set().union(*consumed.values())
    return {
        "unique_days_by_component": {key: len(value) for key, value in consumed.items()},
        "union_unique_days": len(union),
        "landpoints": len({point for point, _, _ in union}),
        "point_years": len({(point, year) for point, year, _ in union}),
        "years": sorted({year for _, year, _ in union}),
    }


def run(args):
    started = time.perf_counter()
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))
    pilot, manifest, extension = map(read, (args.pilot_report, args.prepared_manifest, args.extension))
    source = int(extension["source_update"])
    config = dict(pilot["method"]["configuration"])
    if pilot["status"] != "passed" or pilot["stage"] != "completed" or pilot["completed_updates"] != source:
        raise ValueError("requires completed pilot at the declared source update")
    if manifest["identity"] != pilot["training_plan"]["prepared_manifest_identity"]:
        raise ValueError("extension requires the same prepared TRAIN pool")
    if extension["sampling_seed"] != config["sampling_seed"]:
        raise ValueError("extension must retain the pilot sampling seed")
    config["max_updates"] = int(extension["target_update"])
    plan = build_learning_plan(manifest, config)
    ids = plan["training_artifact_ids"]
    observed = [int(row["artifact_id"]) for row in pilot["updates"]]
    if ids[:source] != observed or observed != pilot["training_plan"]["training_artifact_ids"]:
        raise ValueError("expanded data order does not preserve actual pilot prefix")
    if plan["fit_artifact_ids"] != pilot["training_plan"]["fit_artifact_ids"]:
        raise ValueError("fixed fit panel changed")
    report = {
        "status": "planned", "model_executed": False, "extension": extension,
        "plan": plan, "new_artifact_ids": ids[source:],
        "source_checkpoint": pilot["final_checkpoint"],
        "prefix_matches_actual_updates": True,
        "new_artifacts_overlap_pilot": len(set(ids[source:]) & set(observed)),
        "new_daily_draws": (len(ids)-source)*config["daily_rows_per_update"],
        "new_windows_per_horizon": len(ids)-source,
        "validation_role": "unchanged_held_out_development_not_sealed_test",
        "training_ready": False,
    }
    _atomic_json(args.output, report)
    if args.metadata_only:
        return report

    validation_keys = {(str(row["landpoint_id"]), int(row["year"]))
                       for row in pilot["validation_panel"]["rows"]}
    groups = {name: {key: set() for key in ("daily", "seven_day", "thirty_day")}
              for name in ("pilot", "extension")}
    signatures = None
    for ordinal, identifier in enumerate(ids, start=1):
        artifacts, _ = _load_prepared_artifacts(args.prepared_manifest, [identifier])
        artifact = artifacts[0]
        current_signatures = artifact_signatures(artifact)
        if signatures is None:
            signatures = current_signatures
        elif current_signatures != signatures:
            raise ValueError(f"artifact {identifier} shape/dtype class differs")
        components = artifact.schedule_entry["components"]
        count = int(np.shape(artifact.batch.one_day.inputs.continuous_state)[0])
        if count != config["daily_rows_per_update"]:
            raise ValueError("daily population differs from32")
        if sum(len(row["start_rows"]) for row in components["one_day"]) != count:
            raise ValueError("daily schedule population differs from arrays")
        keys = {"daily": day_keys(components["one_day"])}
        for name, horizon in (("seven_day", 7), ("thirty_day", 30)):
            record = components[name][0]
            keys[name] = day_keys([{**record, "start_rows": record["start_rows"][:1]}],
                                  horizon=horizon)
            days = np.asarray(getattr(artifact.batch, name).inputs.day_index)[0].reshape(-1)
            if days.size != horizon or not np.all(np.diff(days) == 1):
                raise ValueError("nonconsecutive recurrent window")
        point_years = {(point, year) for values in keys.values() for point, year, _ in values}
        if point_years & validation_keys:
            raise ValueError("TRAIN extension overlaps held-out development point-years")
        group = groups["pilot" if ordinal <= source else "extension"]
        for name, values in keys.items():
            group[name].update(values)
        if ordinal % 50 == 0:
            report.update(status="running", checked_artifacts=ordinal,
                          elapsed_seconds=time.perf_counter()-started)
            _atomic_json(args.output, report)
            print(f"checked={ordinal}/{len(ids)} elapsed={report['elapsed_seconds']:.1f}s", flush=True)
    combined = {key: groups["pilot"][key] | groups["extension"][key] for key in groups["pilot"]}
    old_union = set().union(*groups["pilot"].values())
    new_union = set().union(*groups["extension"].values())
    report.update(
        status="passed", coverage={name: summarize(value) for name, value in groups.items()},
        cumulative_coverage=summarize(combined), new_union_days_beyond_pilot=len(new_union-old_union),
        cross_phase_overlap_days=len(new_union & old_union),
        shared_shape_dtype_class=True, held_out_point_year_overlap=0,
        elapsed_seconds=time.perf_counter()-started,
    )
    _atomic_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "cumulative_coverage", "elapsed_seconds")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot-report", "prepared-manifest", "extension", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    run(parser.parse_args())
