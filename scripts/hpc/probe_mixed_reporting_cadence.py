"""Replay saved report events on CPU to compare eager and periodic formatting.

No model/checkpoint payload or Teacher data is loaded. No real training report
is written: the writer formats into memory using the actual production codec.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path


def replay(source, *, interval, count, reporting_class, json_safe, json_value):
    record = copy.deepcopy(source)
    end = int(source["completed_updates"])
    start = max(0, end - count)
    all_updates = source["updates"]
    all_monitors = source.get("monitors", [])
    full_path = Path("memory-only-report.json")
    result = {"full_writes": 0, "progress_writes": 0, "formatted_characters": 0}
    last_full = None

    def memory_writer(path, payload):
        nonlocal last_full
        # Exact formatting path of run_successor_process_head_canonical_gate._atomic_json.
        encoded = json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n"
        result["formatted_characters"] += len(encoded)
        if path == full_path:
            result["full_writes"] += 1
            last_full = encoded
        else:
            result["progress_writes"] += 1

    writer = reporting_class(full_path, write_json=memory_writer,
                             json_safe=json_safe, interval=interval)

    def publish(stage, completed):
        record.update(stage=stage, completed_updates=completed, status="running",
                      next_update=completed + 1)
        record["updates"] = [row for row in all_updates if row["update"] <= completed]
        record["monitors"] = [row for row in all_monitors if row["update"] <= completed]
        writer.publish(record)

    wall, cpu = time.perf_counter(), time.process_time()
    publish("replay_begin", start)
    for step in range(start + 1, end + 1):
        publish("training", step - 1)
        publish("optimizer_completed", step)
        if step % 50 == 0 or step == end:
            # Exercise required report boundaries; no checkpoint is actually saved.
            publish("checkpoint_saving", step)
            publish("checkpoint_saved", step)
        if any(row["update"] == step for row in all_monitors):
            publish("monitor_completed", step)
    publish("completed", end)
    result.update(wall_seconds=time.perf_counter() - wall,
                  cpu_seconds=time.process_time() - cpu,
                  interval=interval, start_update=start, end_update=end,
                  replayed_updates=end - start)
    return result, last_full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=25, choices=(5, 25, 50))
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    from scripts.hpc.mixed_learning_reporting import MixedLearningReporter
    from scripts.hpc.run_successor_process_head_canonical_gate import _json_value
    from scripts.hpc.trained_screen_common import json_safe

    source = json.loads(args.report.read_text(encoding="utf-8"))
    results, finals = {}, {}
    for name, interval in (("eager", 1), ("periodic", 25)):
        print(f"report_cadence stage={name}", flush=True)
        results[name], finals[name] = replay(
            source, interval=interval, count=args.updates,
            reporting_class=MixedLearningReporter, json_safe=json_safe, json_value=_json_value)
    identical = finals["eager"] == finals["periodic"]
    output = {"schema_version": "mixed_reporting_cadence_probe_v1",
              "status": "passed" if identical else "failed",
              "host": platform.node(), "source_root": str(args.source_root.resolve()),
              "source_report": str(args.report.resolve()), "results": results,
              "full_report_identical": identical,
              "formatting_wall_reduction_fraction": 1 - results["periodic"]["wall_seconds"] / results["eager"]["wall_seconds"],
              "model_evaluations": 0, "optimizer_updates": 0,
              "limitations": ["CPU diagnostic replay, not measured training speedup.",
                              "Filesystem write_text/replace omitted; no fsync exists in the actual full writer.",
                              "Saved JSON uses plain types; live NumPy/JAX values may add conversion work.",
                              "Only timing/report stages are replayed, no scientific or optimizer computation."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, separators=(",", ":")), flush=True)
    if not identical:
        raise ValueError("cadence changed complete report payload")


if __name__ == "__main__":
    main()
