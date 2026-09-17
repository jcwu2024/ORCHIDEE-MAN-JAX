"""Subprocess regression for the production continuous dual-device trainer."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_tiny_continuity_probe_covers_serial_resume_sync_and_rejection(tmp_path):
    output = tmp_path / "dual-continuity.json"
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["JAX_COMPILATION_CACHE_DIR"] = str(tmp_path / "jax-cache")
    command = [
        sys.executable,
        "-m",
        "scripts.hpc.probe_dual_training_continuity",
        "--backend",
        "cpu",
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert "dual_training_continuity_probe_v1" in completed.stdout
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert len(report["serial_dual_comparison"]) == 3
    for row in report["serial_dual_comparison"]:
        assert row["gradient_max_abs_error"] <= 2e-5
        assert row["parameter_max_abs_error"] <= 2e-5
        assert row["optimizer_max_abs_error"] <= 2e-5
        assert 0 < row["clip"] < 1
    assert report["checkpoint"]["saved_after_update"] == 2
    assert report["checkpoint"]["resumed_update"] == 3
    assert report["checkpoint"]["parameter_max_abs_error"] <= 2e-5
    assert report["checkpoint"]["optimizer_max_abs_error"] <= 2e-5
    assert report["daily_only_peer_sync_max_abs_error"] <= 2e-5
    assert report["mixed_callback_peer_sync_max_abs_error"] <= 2e-5
    assert report["rejected_failure"]["state_unchanged"] is True
