"""CPU arithmetic and event-order evidence; no real-model speed claims."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc import shared_latent_T7_batch_ad as batch
from scripts.hpc import shared_latent_T7_dispatch_ad as dispatch
from scripts.hpc import probe_shared_latent_T7_dispatch as probe


@pytest.mark.parametrize("mode", dispatch.MODES)
def test_same_complete_ad_mean_and_unreduced_auxiliary(mode):
    def objective(parameters, initial, forcing, mask):
        prediction = parameters["rate"] * initial + forcing
        loss = jnp.sum(jnp.where(mask, prediction**2, 0))
        return loss, {"prediction": prediction, "mask": mask,
                      "hard": {"mask_mismatch": jnp.asarray(0, dtype=jnp.int32)}}

    parameters = {"rate": jnp.asarray(0.6)}
    singles = [(parameters, jnp.asarray([0.2 + index, 0.5 + index]),
                jnp.asarray([0.1, 0.2]) * (index + 1),
                jnp.asarray([True, index % 2 == 0])) for index in range(4)]
    executable = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))
    outputs = [executable(*single) for single in singles]
    merge = jax.jit(lambda *rows: batch.merge_serial_results(jax, rows))
    result, timing = dispatch.execute_four(executable, singles, merge, mode=mode, block=jax.block_until_ready)
    predictions = [0.6 * np.asarray(single[1]) + np.asarray(single[2]) for single in singles]
    expected_loss = np.mean([np.sum(np.where(single[3], prediction**2, 0))
                             for single, prediction in zip(singles, predictions)])
    expected_parameter = np.mean([np.sum(np.where(single[3], 2 * prediction * single[1], 0))
                                  for single, prediction in zip(singles, predictions)])
    expected_states = np.stack([np.where(single[3], 2 * prediction * 0.6, 0)
                                for single, prediction in zip(singles, predictions)]) / 4
    np.testing.assert_allclose(result[0][0], expected_loss, rtol=2e-6)
    np.testing.assert_allclose(result[1][0]["rate"], expected_parameter, rtol=2e-6)
    np.testing.assert_allclose(result[1][1], expected_states, rtol=2e-6)
    np.testing.assert_array_equal(result[0][1]["mask"], np.stack([single[3] for single in singles]))
    np.testing.assert_allclose(result[0][1]["prediction"], predictions, rtol=2e-6)
    np.testing.assert_array_equal(result[0][1]["hard"]["mask_mismatch"], np.zeros(4))
    # CPU elapsed values are deliberately not accepted as real-model speed.
    assert timing["timing_includes_merge_and_final_completion"]
    assert len(timing["sample_seconds"]) == 4
    jax.block_until_ready(outputs)


@pytest.mark.parametrize("mode", dispatch.MODES)
def test_exact_sync_protocol_includes_initial_and_merge_dependencies(mode):
    events = []
    singles = [(index,) for index in range(4)]

    def executable(index):
        events.append("call" + str(index))
        return "result" + str(index)

    def block(value):
        events.append("input_block" if isinstance(value, tuple) else "block:" + value)

    def merge(*rows):
        assert rows == tuple("result" + str(index) for index in range(4))
        events.append("merge")
        return "merged"

    clock_values = iter(range(20))
    result, timing = dispatch.execute_four(executable, singles, merge, mode=mode, block=block, clock=lambda: next(clock_values))
    expected = ["input_block"]
    for index in range(4):
        expected.append("call" + str(index))
        if mode == "block_each":
            expected.append("block:result" + str(index))
    assert events == expected + ["merge", "block:merged"]
    assert result == "merged"
    assert timing["sample_time_kind"] == ("synchronized_full_AD" if mode == "block_each" else "host_dispatch_only")


def test_pair_orders_and_invalid_cases():
    assert dispatch.alternating_orders() == [dispatch.MODES, dispatch.MODES[::-1], dispatch.MODES]
    with pytest.raises(ValueError, match="three"):
        dispatch.alternating_orders(2)
    with pytest.raises(ValueError, match="four"):
        dispatch.execute_four(None, [()], None, mode="dispatch_four", block=None)


def test_new_independent_scope_rejects_old_or_unapproved_budget():
    scope = {"schema_version": "latent_T7_dispatch_scope_v1", "approved": True,
             "funding_scope": "latent_T7_dispatch_20261009", "gpu_count": 1,
             "limit_gpu_seconds": 900, "session_limit_seconds": 900,
             "model_compile_cap": 0, "optimizer_update_cap": 0, "pairs": 3, "output": "new.json"}
    dispatch.validate_scope(scope, output=Path("new.json"), wall_seconds=840)
    for changes in ({"approved": False}, {"funding_scope": "latent_B4_T7_20261008"},
                    {"limit_gpu_seconds": 7200}, {"model_compile_cap": 1}, {"output": "old.json"}):
        with pytest.raises(ValueError):
            dispatch.validate_scope({**scope, **changes}, output=Path("new.json"), wall_seconds=840)
    with pytest.raises(ValueError, match="exit"):
        dispatch.validate_scope(scope, output=Path("new.json"), wall_seconds=841)


def test_saved_only_cli_dry_run_requires_no_gpu_or_model(monkeypatch):
    from types import SimpleNamespace
    args = SimpleNamespace(horizon=7, original_executable=None, candidate_executable=Path("saved.pkl"),
        validate_only=False, pairs=3, wall_seconds=840, load_seconds=360, call_seconds=180,
        output=Path("T7-dispatch-new.json"), approval_report=Path("new-approval.json"), dry_run=True)
    monkeypatch.setattr(probe.pair, "assets", lambda *_args: pytest.fail("dry-run loaded model"))
    plan = probe.run(args)
    assert plan["model_compiles"] == {"B1": 0, "B4": 0}
    assert plan["GPU_executed"] is False
    assert plan["orders"] == [["block_each", "dispatch_four"], ["dispatch_four", "block_each"], ["block_each", "dispatch_four"]]


def test_completed_numeric_failure_keeps_report_and_budget_finally(monkeypatch, tmp_path):
    """CPU child stub exercises the actual existing guardian settlement path."""
    from contextlib import contextmanager
    import json
    from types import SimpleNamespace
    from scripts.hpc import shared_gpu_budget as budget

    report = {"status": "numerical_failed", "execution_completed": True,
              "GPU_executed": True, "adoption": False, "scientific_acceptance": False}
    report_path = tmp_path / "preserved-report.json"
    ledger = {"limit_gpu_seconds": 900, "session_limit_seconds": 900, "sessions": []}

    @contextmanager
    def locked(_path):
        yield ledger

    class Child:
        pid = 100001

        def wait(self, timeout):
            assert timeout > 0
            report_path.write_text(json.dumps(report))
            return probe.result_exit_code(report)

    def absent_group(*_args):
        raise ProcessLookupError

    monkeypatch.setattr(budget, "LIMIT", 900)
    monkeypatch.setattr(budget, "SESSION_LIMIT", 900)
    monkeypatch.setattr(budget, "locked", locked)
    monkeypatch.setattr(budget.subprocess, "Popen", lambda *_args, **_kwargs: Child())
    monkeypatch.setattr(budget.os, "killpg", absent_group, raising=False)
    monkeypatch.setattr(budget.os, "uname", lambda: SimpleNamespace(nodename="CPU-stub"), raising=False)
    monkeypatch.setattr(budget.signal, "signal", lambda *_args: None)
    clock_steps = iter(index / 10 for index in range(100))
    monkeypatch.setattr(budget.time, "monotonic", lambda: next(clock_steps))
    args = SimpleNamespace(seconds=20, cards=1, command=["--", "CPU-child-stub"],
        ledger=tmp_path / "ledger.json", stage="saved-dispatch", funding_scope=None,
        approved_limit_gpu_seconds=None, approval_note=None)
    assert budget.run(args) == 3
    assert json.loads(report_path.read_text()) == report
    assert ledger["sessions"][0]["status"] == "finished"
    assert ledger["sessions"][0]["exit_status"] == 3
    assert ledger["sessions"][0]["within_session_cap"]
    assert ledger["charged_gpu_seconds"] > 0
    assert ledger["remaining_gpu_seconds"] < 900
    assert probe.result_exit_code({"status": "passed", "GPU_executed": True}) == 0
    assert probe.result_exit_code({"status": "resource_failed", "GPU_executed": False}) == 1


def test_final_stdout_receipt_does_not_serialize_full_numpy_metadata():
    import json
    report = {"status": "numerical_failed", "execution_completed": True,
        "adoption": False, "scientific_acceptance": False,
        "full_comparisons": {"numpy_coordinate": np.int64(434)}}
    receipt = probe.final_receipt(report, Path("saved.json"))
    assert json.loads(json.dumps(receipt))["process_exit_code"] == 3
    assert receipt["execution_completed"] is True
    assert "full_comparisons" not in receipt
