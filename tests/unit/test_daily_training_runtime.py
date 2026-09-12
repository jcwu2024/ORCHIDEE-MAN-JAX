from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from scripts.hpc import check_daily_training_runtime as runtime_check

jax.config.update("jax_enable_x64", True)


def test_runtime_check_exercises_public_streamed_gradients_and_weighted_adamw():
    report = runtime_check.check_runtime(jax, jnp)

    assert report["schema_version"] == "daily_training_runtime_check_v1"
    assert report["status"] == "passed"
    assert report["scope"] == "synthetic_runtime_interfaces_only"
    assert report["real_model_checked"] is False
    assert report["teacher_data_read"] is False
    assert report["scientific_acceptance"] is False
    assert report["real_training_updates"] == 0
    assert report["numpy_version"]

    rollout = report["rollout"]
    assert rollout["horizons"] == [1, 2, 3, 7]
    assert rollout["dtypes"] == ["float32", "float64"]
    assert len(rollout["direct_streamed_cases"]) == 8
    assert all(row["passed"] for row in rollout["direct_streamed_cases"])
    assert all(
        row["reverse_compile_count"] == 1
        for row in rollout["direct_streamed_cases"]
    )
    assert all(row["passed"] for row in rollout["scalar_float0_aot"])
    assert all(row["passed"] for row in rollout["fraction_state_coupling"])
    assert all(
        row["passed"] for row in rollout["same_shape_target_context_reuse"]
    )
    assert all(
        row["reverse_compile_count_before"]
        == row["reverse_compile_count_after"]
        == 1
        for row in rollout["same_shape_target_context_reuse"]
    )

    optimizer = report["microbatch_adamw"]
    assert optimizer["microbatch_counts"] == [2, 3]
    assert optimizer["direct_batch_count"] == 5
    assert optimizer["optimizer_step"] == 1
    assert optimizer["parameters_and_optimizer_finite"]
    assert optimizer["parameters_changed"]
    assert optimizer["passed"]


def test_cli_defaults_to_gpu_ignores_wrapper_arguments_and_returns_nonzero_on_failure(
    tmp_path, monkeypatch
):
    output = tmp_path / "runtime.json"
    observed = {}

    def fail(backend):
        observed["backend"] = backend
        raise RuntimeError("simulated runtime failure")

    monkeypatch.setattr(runtime_check, "_initialize_runtime", fail)
    result = runtime_check.main(
        [
            "--output",
            str(output),
            "--dataset-manifest",
            "ignored.json",
            "--prepared-manifest",
            "also-ignored.json",
        ]
    )

    assert result == 1
    assert observed["backend"] == "gpu"
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "daily_training_runtime_check_v1"
    assert report["status"] == "failed"
    assert report["passed"] is False
    assert report["requested_backend"] == "gpu"
    assert report["real_model_checked"] is False
    assert "simulated runtime failure" in report["error"]
