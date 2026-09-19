"""Bounded progress writes and periodic complete mixed-training reports.

This module never touches model state, gradients, or sampling. Full reports
retain the existing schema/history; the small sidecar is for live progress.
"""
from __future__ import annotations

from pathlib import Path


class MixedLearningReporter:
    """Publish full reports at boundaries, compact progress at every event.

    The caller must emit ``checkpoint_saving`` before replacing any checkpoint
    and ``checkpoint_saved`` afterwards. Thus even a crash between checkpoint
    replacement and the second report write leaves all matching update records.
    A crash between regular flushes can lose telemetry since the last flush;
    recovery rolls back to the last checkpoint, as before.
    """

    HOT_STAGES = frozenset(("training", "optimizer_completed"))

    def __init__(self, output, *, write_json, json_safe, interval=25):
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("report interval must be a positive integer")
        self.output = Path(output)
        self.progress_path = self.output.with_name(self.output.stem + ".progress.json")
        self.write_json, self.json_safe = write_json, json_safe
        self.interval = interval
        self.full_write_count = 0
        self.last_full_update = None
        self.last_full_elapsed = None

    @property
    def policy(self):
        return {
            "schema_version": "mixed_learning_reporting_v1",
            "full_report_interval_updates": self.interval,
            "progress_path": str(self.progress_path),
            "checkpoint_order": "full_report_before_and_after_checkpoint",
            "full_history_retained": True,
            "abrupt_exit_telemetry": "since_last_full_flush_may_be_lost; resume_from_checkpoint",
        }

    def publish(self, report, *, force=False):
        stage = report["stage"]
        completed = report["completed_updates"]
        full = (force or self.interval == 1 or self.last_full_update is None
                or stage not in self.HOT_STAGES
                or (stage == "optimizer_completed" and completed % self.interval == 0))
        if full:
            # Advance persisted metadata only after the atomic writer succeeds.
            self.write_json(self.output, self.json_safe(report))
            self.last_full_update = completed
            self.last_full_elapsed = report.get("elapsed_seconds")
            self.full_write_count += 1

        progress = {
            "schema_version": "mixed_learning_progress_v1",
            "full_report_path": str(self.output),
            "full_report_completed_updates": self.last_full_update,
            "full_report_elapsed_seconds": self.last_full_elapsed,
            "latest_recorded_monitor_update": max(
                (row["update"] for row in report.get("monitors", [])), default=None),
        }
        for key in ("status", "stage", "completed_updates",
                    "elapsed_seconds", "training_seconds", "phase_training_seconds",
                    "data_consumption", "learning_progress", "memory", "host_peak_bytes",
                    "scientific_acceptance", "error"):
            if key in report:
                progress[key] = report[key]
        if stage == "training" and "next_update" in report:
            progress["next_update"] = report["next_update"]
        if (stage == "monitoring" or stage.startswith("monitor_")) and stage != "monitor_completed":
            if "monitor_update" in report:
                progress["monitor_update"] = report["monitor_update"]
        if report.get("updates"):
            latest = report["updates"][-1]
            progress["latest_update"] = {key: latest[key] for key in (
                "update", "daily_loss", "T7", "T30", "mixed_loss", "norm", "clip",
                "seconds", "objective", "phase") if key in latest}
        if report.get("checkpoints"):
            progress["latest_checkpoint"] = report["checkpoints"][-1]
        self.write_json(self.progress_path, self.json_safe(progress))
        return full
