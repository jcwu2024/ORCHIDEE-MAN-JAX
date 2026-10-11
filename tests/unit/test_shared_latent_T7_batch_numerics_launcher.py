"""CPU-only admission and lifecycle coverage for the bounded GPU launcher."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from scripts.hpc import probe_shared_latent_T7_batch_numerics as probe
from scripts.hpc import run_shared_latent_T7_batch_numerics as launcher


ROOT = Path(__file__).resolve().parents[2]
REAL_SOURCE = ROOT / "scripts" / "hpc" / "capture_shared_latent_T7_summary_operands.py"


def captured_bundle(receipt_sha):
    x = np.zeros((4, 257, 4), dtype=np.float32)
    x[:, ::16, :] = 0.25
    direction = np.zeros_like(x)
    direction[3, 3, 2] = 1
    kernel = np.zeros((4, 384), dtype=np.float32)
    kernel[range(4), range(4)] = 1
    bias = np.zeros((384,), dtype=np.float32)
    cotangent = np.zeros((4, 257, 384), dtype=np.float32)
    cotangent[:, ::16, :] = 0.5
    return {
        "x": x,
        "kernel": kernel,
        "bias": bias,
        "direction": direction,
        "cotangent": cotangent,
        "mask": np.ones((4, 257, 384), dtype=bool),
        "metadata": {
            "schema_version": probe.SCHEMA,
            "origin": "captured_component_operands",
            "parameter_file_sha256": probe.PARAMETERS_SHA256,
            "prepared_context_sha256": probe.CONTEXT_SHA256,
            "artifact_ids": probe.ARTIFACT_IDS,
            "parameter_path": "fixture-only",
            "capture_boundary": "test fixture",
            "capture_source_sha256": launcher.CAPTURE_SHA,
            "capture_receipt_sha256": receipt_sha,
            "direction_coordinate": 434,
            "direction_sample": 3,
            "window_index": 0,
            "day_index": 0,
            "mask_meaning": "test fixture",
        },
    }


def prepare_args(tmp_path, monkeypatch):
    scope_path = tmp_path / "scope.json"
    output = tmp_path / "result.json"
    ledger = tmp_path / "one-shot.json"
    operands = tmp_path / "operands.npz"
    receipt_path = tmp_path / "capture.json"
    capture_source = tmp_path / REAL_SOURCE.name
    capture_source.write_bytes(REAL_SOURCE.read_bytes())
    bundle = captured_bundle("fixture-receipt-pending")
    identities = {
        key: {"shape": list(bundle[key].shape), "dtype": str(bundle[key].dtype),
              "sha256_bytes_C_order": hashlib.sha256(bundle[key].tobytes(order="C")).hexdigest()}
        for key in probe.ARRAY_KEYS
    }
    receipt = {
        "status": "operands_captured",
        "array_identities": identities,
        "current_execution": {
            "git_head": "fixture-head",
            "source_files": {"scripts/hpc/probe_shared_latent_T7_batch_numerics.py": {
                "sha256": launcher.CAPTURE_PROBE_SHA}},
        },
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    bundle["metadata"]["capture_receipt_sha256"] = receipt_sha
    bundle = probe.validate_operands(bundle)
    probe.save_operands(operands, bundle)
    monkeypatch.setattr(launcher, "OPERAND_SHA", hashlib.sha256(operands.read_bytes()).hexdigest())
    monkeypatch.setattr(launcher, "RECEIPT_SHA", receipt_sha)
    scope = {
        "schema_version": launcher.SCHEMA,
        "scope_id": launcher.SCOPE_ID,
        "caps": launcher.CAPS,
        "node": "gln01",
        "cards": 1,
        "nonexclusive": True,
        "worst_case_cost_cny": 0,
        "repeats": 3,
        "status": "proposed",
        "gpu_host_index": 0,
        "gpu_uuid": "GPU-test-fixture",
        "ledger": str(ledger),
        "output": str(output),
    }
    scope_path.write_text(json.dumps(scope), encoding="utf-8")
    args = launcher.parser().parse_args([
        "--checkout-root", str(ROOT), "--operands", str(operands),
        "--capture-receipt", str(receipt_path), "--capture-source", str(capture_source),
        "--output", str(output), "--scope-json", str(scope_path), "--ledger", str(ledger),
        "--dry-run",
    ])
    return args, scope, scope_path, output, ledger


def minimal_plan(tmp_path):
    output = tmp_path / "result.json"
    scope = {
        "caps": dict(launcher.CAPS),
        "gpu_host_index": 0,
        "gpu_uuid": "GPU-test-fixture",
    }
    plan = {
        "scope": scope,
        "commands": {"preflight": ["fake-preflight"], "probe": ["fake-probe"]},
        "output": str(output),
        "ledger": str(tmp_path / "one-shot.json"),
        "checkout": str(ROOT),
        "preflight_output": str(output.with_suffix(".runtime-preflight.json")),
        "shell_started_epoch": None,
    }
    for key in ("output", "ledger", "preflight_output"):
        assert Path(plan[key]).resolve().is_relative_to(tmp_path.resolve())
    return plan


class FakeGroups:
    def __init__(self, *, live=()):
        self.live = set(live)
        self.killed = []

    def owner(self, pid, groups):
        return pid in groups

    def kill(self, group, signum):
        self.killed.append((group, signum))
        self.live.discard(group)

    def alive(self, group):
        return group in self.live


class FakeMonitor:
    def __init__(self, *rows):
        self.rows = list(rows)
        self.calls = 0

    def sample(self):
        self.calls += 1
        if self.rows:
            return self.rows.pop(0)
        return {"memory_mib": 0, "utilization_percent": 0, "processes": []}


def gpu_row(memory=0, utilization=0, processes=()):
    return {"memory_mib": memory, "utilization_percent": utilization,
            "processes": list(processes)}


def test_prepare_binds_capture_fixture_and_dry_run_is_nonexecuting(tmp_path, monkeypatch, capsys):
    args, _, _, _, _ = prepare_args(tmp_path, monkeypatch)
    monkeypatch.setattr(launcher, "Supervisor", lambda *a, **k: pytest.fail("dry-run started supervisor"))
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *a, **k: pytest.fail("dry-run checked runtime HEAD"))
    assert launcher.main([
        "--checkout-root", str(args.checkout_root), "--operands", str(args.operands),
        "--capture-receipt", str(args.capture_receipt), "--capture-source", str(args.capture_source),
        "--output", str(args.output), "--scope-json", str(args.scope_json), "--ledger", str(args.ledger),
        "--dry-run",
    ]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["identities"] == {
        "operand": launcher.OPERAND_SHA,
        "receipt": launcher.RECEIPT_SHA,
        "capture_source": launcher.CAPTURE_SHA,
    }
    assert planned["planned_measurements"] == 12
    assert planned["planned_arrays"] == 91
    assert planned["planned_small_compiles"] == 4
    assert planned["GPU_executed"] is False


def test_prepare_rejects_a_different_checkout_root_before_asset_admission(tmp_path, monkeypatch):
    args, _, _, _, _ = prepare_args(tmp_path, monkeypatch)
    args.checkout_root = tmp_path
    with pytest.raises(ValueError, match="own declared checkout"):
        launcher.prepare(args)


@pytest.mark.parametrize("bad_asset", ["operands", "receipt", "source"])
@pytest.mark.parametrize("missing", [False, True])
def test_prepare_rejects_missing_or_invalid_capture_assets(tmp_path, monkeypatch, bad_asset, missing):
    args, _, _, _, _ = prepare_args(tmp_path, monkeypatch)
    path = getattr(args, {"operands": "operands", "receipt": "capture_receipt", "source": "capture_source"}[bad_asset])
    assert path.resolve().is_relative_to(tmp_path.resolve())
    if missing:
        path.unlink()
    else:
        path.write_text("invalid replacement", encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        launcher.prepare(args)


@pytest.mark.parametrize("identity", ["OPERAND_SHA", "RECEIPT_SHA", "CAPTURE_SHA"])
def test_prepare_rejects_wrong_expected_asset_hash(tmp_path, monkeypatch, identity):
    args, _, _, _, _ = prepare_args(tmp_path, monkeypatch)
    monkeypatch.setattr(launcher, identity, "0" * 64)
    with pytest.raises(ValueError, match="identity differs"):
        launcher.prepare(args)


@pytest.mark.parametrize("status", ["closed", "open"])
def test_prepare_rejects_closed_or_repeated_one_shot_scope(tmp_path, monkeypatch, status):
    args, scope, scope_path, _, ledger = prepare_args(tmp_path, monkeypatch)
    if status == "closed":
        assert scope_path.resolve().is_relative_to(tmp_path.resolve())
        scope["status"] = "closed"
        scope_path.write_text(json.dumps(scope), encoding="utf-8")
    else:
        assert ledger.resolve().is_relative_to(tmp_path.resolve())
        ledger.write_text("already reserved", encoding="utf-8")
    with pytest.raises((ValueError, FileExistsError), match="scope|reserved|closed"):
        launcher.prepare(args)


def test_prepare_rejects_outer_timeout_marker(tmp_path, monkeypatch):
    args, _, _, output, _ = prepare_args(tmp_path, monkeypatch)
    marker = output.with_suffix(".outer-timeout.json")
    assert marker.resolve().is_relative_to(tmp_path.resolve())
    marker.write_text('{"status":"deadline"}', encoding="utf-8")
    with pytest.raises(FileExistsError, match="existing evidence"):
        launcher.prepare(args)


def test_live_prepare_rejects_unapproved_hostname_before_remote_path_gate(tmp_path, monkeypatch):
    args, scope, scope_path, _, _ = prepare_args(tmp_path, monkeypatch)
    assert scope_path.resolve().is_relative_to(tmp_path.resolve())
    scope.update(status="open", execution_authorized=True, approval_note="fixture only")
    scope_path.write_text(json.dumps(scope), encoding="utf-8")
    args.dry_run = False
    monkeypatch.setattr(launcher, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(launcher.socket, "gethostname", lambda: "cln01")
    with pytest.raises(ValueError, match="gln01 GPU test node"):
        launcher.prepare(args)


def test_live_host_gate_allows_only_declared_posix_gpu_nodes():
    launcher.validate_live_host("gln01", "posix")
    launcher.validate_live_host("ibc13b02n04.cluster", "posix")
    with pytest.raises(ValueError, match="GPU test node"):
        launcher.validate_live_host("cln01", "posix")
    with pytest.raises(ValueError, match="GPU test node"):
        launcher.validate_live_host("gln01", "nt")


def test_frozen_source_manifest_requires_exact_paths_and_hashes():
    source_files = {path: "fixture-sha-" + str(index)
                    for index, path in enumerate(launcher.SOURCE_PATHS)}
    launcher.validate_source_manifest({"source_files": source_files}, source_files)
    poisoned = dict(source_files)
    poisoned[launcher.SOURCE_PATHS[0]] = "changed"
    with pytest.raises(ValueError, match="frozen scope manifest"):
        launcher.validate_source_manifest({"source_files": source_files}, poisoned)
    missing = dict(source_files)
    missing.pop(launcher.SOURCE_PATHS[-1])
    with pytest.raises(ValueError, match="frozen scope manifest"):
        launcher.validate_source_manifest({"source_files": source_files}, missing)


def test_main_rejects_dirty_checkout_before_supervisor(tmp_path, monkeypatch, capsys):
    plan = minimal_plan(tmp_path)
    plan["scope"]["code_git_head"] = "frozen-head"
    monkeypatch.setattr(launcher, "prepare", lambda args: plan)
    checks = iter(["frozen-head\n", " M tracked.py\n"])
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *a, **k: next(checks))
    monkeypatch.setattr(launcher, "Supervisor", lambda *a, **k: pytest.fail("dirty checkout reached supervisor"))
    assert launcher.main([
        "--checkout-root", str(ROOT), "--operands", "unused.npz", "--capture-receipt", "unused.json",
        "--capture-source", "unused.py", "--output", str(tmp_path / "result.json"),
        "--scope-json", "unused-scope.json", "--ledger", str(tmp_path / "ledger.json"),
    ]) == 2
    assert "clean frozen checkout" in capsys.readouterr().out


def test_supervisor_success_requires_both_reports_and_clean_release(tmp_path, monkeypatch):
    plan = minimal_plan(tmp_path)
    monitor = FakeMonitor(gpu_row(), gpu_row())
    supervisor = launcher.Supervisor(plan, monitor=monitor, groups=FakeGroups(), watchdog=False)

    def passed_phase(self, name, seconds):
        if name == "preflight":
            Path(self.plan["preflight_output"]).write_text(json.dumps({
                "passed": True, "actual_backend": "gpu", "jax_version": "0.4.38", "devices": ["GPU"]
            }), encoding="utf-8")
        else:
            Path(self.plan["output"]).write_text(json.dumps({
                "status": "passed", "passed": True, "backend": "gpu", "jax_version": "0.4.38",
                "operand_sha256": launcher.OPERAND_SHA, "measurement_count": 12,
                "preserved_array_count": 91, "total_small_graph_compiles": 4,
                "tolerances": {"rtol": 1e-5, "atol": 1e-6},
                "results_container": str(Path(self.plan["output"]).with_suffix(".results.npz")),
            }), encoding="utf-8")
            Path(self.plan["output"]).with_suffix(".results.npz").write_bytes(b"fixture")
        return "passed", 0

    monkeypatch.setattr(launcher.Supervisor, "phase", passed_phase)
    assert supervisor.run() == 0
    ledger = json.loads(Path(plan["ledger"]).read_text(encoding="utf-8"))
    record = json.loads(Path(plan["output"]).with_suffix(".supervisor.json").read_text(encoding="utf-8"))
    assert ledger["status"] == "closed" and ledger["charged_gpu_seconds"] <= 1200
    assert record["primary_status"] == "passed" and record["release_verified"] is True


def test_numerical_failure_exit_three_remains_primary(tmp_path, monkeypatch):
    plan = minimal_plan(tmp_path)
    supervisor = launcher.Supervisor(plan, monitor=FakeMonitor(gpu_row(), gpu_row()),
                                     groups=FakeGroups(), watchdog=False)
    monkeypatch.setattr(launcher.Supervisor, "phase", lambda self, name, seconds: ("numerical_failed", 3))
    assert supervisor.run() == 3
    record = json.loads(Path(plan["output"]).with_suffix(".supervisor.json").read_text(encoding="utf-8"))
    ledger = json.loads(Path(plan["ledger"]).read_text(encoding="utf-8"))
    assert record["primary_status"] == ledger["primary_status"] == "numerical_failed"
    assert record["exit_code"] == ledger["exit_code"] == 3


@pytest.mark.parametrize("row", [gpu_row(memory=1), gpu_row(utilization=1)])
def test_initial_gpu_observation_requires_zero_memory_and_utilization(tmp_path, row):
    plan = minimal_plan(tmp_path)
    supervisor = launcher.Supervisor(plan, monitor=FakeMonitor(row, gpu_row()),
                                     groups=FakeGroups(), watchdog=False)
    assert supervisor.run() == 125
    record = json.loads(Path(plan["output"]).with_suffix(".supervisor.json").read_text(encoding="utf-8"))
    assert record["primary_status"] == "gpu_not_idle"
    assert record["phases"] == [] and record["release_verified"] is True


def test_runtime_failure_and_monitor_timeout_close_without_false_pass(tmp_path, monkeypatch):
    plan = minimal_plan(tmp_path / "runtime")
    Path(plan["output"]).parent.mkdir(parents=True, exist_ok=True)
    supervisor = launcher.Supervisor(plan, monitor=FakeMonitor(gpu_row(), gpu_row()),
                                     groups=FakeGroups(), watchdog=False)
    monkeypatch.setattr(launcher.Supervisor, "phase",
                        lambda self, name, seconds: (_ for _ in ()).throw(RuntimeError("child failed to start")))
    assert supervisor.run() == 1
    record = json.loads(Path(plan["output"]).with_suffix(".supervisor.json").read_text(encoding="utf-8"))
    assert record["primary_status"] == "runtime_failed" and record["release_verified"]

    plan2 = minimal_plan(tmp_path / "monitor-timeout")
    Path(plan2["output"]).parent.mkdir(parents=True, exist_ok=True)
    timeout_monitor = SimpleNamespace(sample=lambda: (_ for _ in ()).throw(subprocess.TimeoutExpired("nvidia-smi", 4)))
    supervisor2 = launcher.Supervisor(plan2, monitor=timeout_monitor, groups=FakeGroups(), watchdog=False)
    assert supervisor2.run() == 124
    record2 = json.loads(Path(plan2["output"]).with_suffix(".supervisor.json").read_text(encoding="utf-8"))
    assert record2["primary_status"] == "deadline_exceeded" and record2["exit_code"] == 124
    assert not record2["release_verified"]


def test_deadline_elapsed_during_child_startup_stops_before_sampling(tmp_path):
    plan = minimal_plan(tmp_path)
    now = [0.0]
    groups = FakeGroups()

    class Process:
        pid = 321
        def poll(self):
            return None
        def wait(self, timeout=None):
            return None

    def slow_start(*args, **kwargs):
        now[0] = 1021.0  # Work deadline is 1020 seconds; total reservation is 1200.
        return Process()

    monitor = FakeMonitor(gpu_row(), gpu_row())
    supervisor = launcher.Supervisor(plan, monitor=monitor, clock=lambda: now[0], sleeper=lambda _: None,
                                     popen=slow_start, groups=groups, watchdog=False)
    status = supervisor.phase("preflight", 240)
    assert status == ("deadline_exceeded", 124)
    assert monitor.calls == 0
    assert supervisor.record["phases"][0]["elapsed_seconds"] == 1021.0


def test_memory_limit_and_foreign_process_never_signal_foreign_pid(tmp_path):
    plan = minimal_plan(tmp_path / "memory")
    Path(plan["output"]).parent.mkdir(parents=True, exist_ok=True)
    monitor = FakeMonitor(gpu_row(), gpu_row(memory=24577), gpu_row())
    groups = FakeGroups(live={456})
    child = SimpleNamespace(pid=456, poll=lambda: None, wait=lambda timeout=None: None)
    supervisor = launcher.Supervisor(plan, monitor=monitor, groups=groups,
                                     popen=lambda *a, **k: child, watchdog=False)
    assert supervisor.run() == 137

    plan2 = minimal_plan(tmp_path / "foreign")
    Path(plan2["output"]).parent.mkdir(parents=True, exist_ok=True)
    groups2 = FakeGroups()
    monitor2 = FakeMonitor(gpu_row(processes=[{"pid": 999, "memory_mib": 8, "name": "other"}]), gpu_row())
    supervisor2 = launcher.Supervisor(plan2, monitor=monitor2, groups=groups2, watchdog=False)
    assert supervisor2.run() == 125
    assert all(group != 999 for group, _ in groups2.killed)


def test_child_cleanup_kills_only_owned_group_and_unverified_release_is_not_pass(tmp_path, monkeypatch):
    plan = minimal_plan(tmp_path)
    Path(plan["output"]).parent.mkdir(parents=True, exist_ok=True)
    now = [0.0]
    groups = FakeGroups(live={123})

    class StuckProcess:
        pid = 123
        def poll(self):
            return None
        def wait(self, timeout=None):
            return None

    monitor = FakeMonitor(gpu_row(), gpu_row(memory=24577), gpu_row())
    supervisor = launcher.Supervisor(plan, monitor=monitor, clock=lambda: now[0],
                                     sleeper=lambda duration: now.__setitem__(0, now[0] + duration),
                                     popen=lambda *a, **k: StuckProcess(), groups=groups, watchdog=False)
    assert supervisor.sample(initial=True) is None
    assert supervisor.phase("preflight", 240) == ("memory_exceeded", 137)
    assert [group for group, _ in groups.killed] == [123]
    assert supervisor.owned == set() and supervisor.group_history == [123]
    assert supervisor.cleanup()
    assert [group for group, _ in groups.killed] == [123]

    plan2 = minimal_plan(tmp_path / "release-unknown")
    Path(plan2["output"]).parent.mkdir(parents=True, exist_ok=True)
    supervisor2 = launcher.Supervisor(plan2, monitor=FakeMonitor(gpu_row()), groups=FakeGroups(), watchdog=False)
    def passed_phase(self, name, seconds):
        if name == "preflight":
            Path(self.plan["preflight_output"]).write_text(json.dumps({
                "passed": True, "actual_backend": "gpu", "jax_version": "0.4.38", "devices": ["GPU"]
            }), encoding="utf-8")
        else:
            result_path = Path(self.plan["output"]).with_suffix(".results.npz")
            result_path.write_bytes(b"fixture")
            Path(self.plan["output"]).write_text(json.dumps({
                "status": "passed", "passed": True, "backend": "gpu", "jax_version": "0.4.38",
                "operand_sha256": launcher.OPERAND_SHA, "measurement_count": 12,
                "preserved_array_count": 91, "total_small_graph_compiles": 4,
                "tolerances": {"rtol": 1e-5, "atol": 1e-6}, "results_container": str(result_path),
            }), encoding="utf-8")
        return "passed", 0
    monkeypatch.setattr(launcher.Supervisor, "phase", passed_phase)
    monkeypatch.setattr(supervisor2, "cleanup", lambda: False)
    assert supervisor2.run() == 125
    record = json.loads(Path(plan2["output"]).with_suffix(".supervisor.json").read_text(encoding="utf-8"))
    assert record["primary_status"] == "passed" and record["release_verified"] is False
    assert record["status"] == "resource_unverified" and record["exit_code"] == 125


def test_primary_numerical_failure_and_unknown_release_are_recorded_separately(tmp_path, monkeypatch):
    plan = minimal_plan(tmp_path)
    supervisor = launcher.Supervisor(plan, monitor=FakeMonitor(gpu_row(), gpu_row(memory=2)),
                                     groups=FakeGroups(), watchdog=False)
    monkeypatch.setattr(launcher.Supervisor, "phase", lambda self, name, seconds: ("numerical_failed", 3))
    assert supervisor.run() == 3
    record = json.loads(Path(plan["output"]).with_suffix(".supervisor.json").read_text(encoding="utf-8"))
    ledger = json.loads(Path(plan["ledger"]).read_text(encoding="utf-8"))
    assert record["primary_status"] == ledger["primary_status"] == "numerical_failed"
    assert record["release_verified"] is False and record["exit_code"] == 3
    assert ledger["status"] == "closed_release_unverified" and ledger["charged_gpu_seconds"] == 1200


def test_monitor_uuid_identity_and_timeout_are_bounded(monkeypatch):
    monitor = launcher.NvidiaMonitor(0, "GPU-expected", deadline=10, clock=lambda: 0)
    outputs = iter([
        "0, GPU-other, Tesla V100, 1, 0\n",
        "GPU-expected, 4, 1, python\n",
    ])
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=next(outputs), stderr=""))
    with pytest.raises(ValueError, match="identity"):
        monitor.sample()

    monkeypatch.setattr(launcher.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("nvidia-smi", 4)))
    with pytest.raises(subprocess.TimeoutExpired):
        monitor.command(["nvidia-smi"])


def small_bundle():
    x = np.arange(16, dtype=np.float32).reshape(4, 2, 2) / 10
    direction = np.zeros_like(x)
    direction[3, 0, 0] = 1
    return {
        "x": x,
        "kernel": np.eye(2, dtype=np.float32),
        "bias": np.zeros((2,), dtype=np.float32),
        "direction": direction,
        "cotangent": np.ones((4, 2, 2), dtype=np.float32),
        "mask": np.ones((4, 2, 2), dtype=bool),
        "metadata": {"schema_version": probe.SCHEMA, "origin": "synthetic_method_fixture"},
    }


def test_partial_snapshots_atomically_preserve_full_arrays_and_nan(tmp_path):
    bundle = small_bundle()
    raw = {
        "primal": np.arange(16, dtype=np.float32).reshape(4, 2, 2),
        "tangent": np.full((4, 2, 2), np.nan, dtype=np.float32),
        "loss": np.asarray(np.nan),
        "parameter_gradient": {"kernel": np.ones((2, 2), dtype=np.float32),
                               "bias": np.full((2,), np.nan, dtype=np.float32)},
    }
    output = tmp_path / "partial.npz"
    probe.save_partial_measurements(output, bundle, [{"cycle": 0, "route": "b1", "result": raw}])
    with np.load(output, allow_pickle=False) as arrays:
        assert set(arrays.files) == {
            "cycle0.b1.primal", "cycle0.b1.tangent", "cycle0.b1.loss",
            "cycle0.b1.parameter_gradient.kernel", "cycle0.b1.parameter_gradient.bias",
            "oracle.primal", "oracle.tangent", "oracle.loss",
            "oracle.parameter_gradient.kernel", "oracle.parameter_gradient.bias",
            "oracle.input_gradient", "oracle.mask",
        }
        np.testing.assert_array_equal(arrays["cycle0.b1.primal"], raw["primal"])
        assert np.isnan(arrays["cycle0.b1.tangent"]).all()
        assert np.isnan(arrays["cycle0.b1.loss"])
        np.testing.assert_array_equal(arrays["cycle0.b1.parameter_gradient.bias"], raw["parameter_gradient"]["bias"])
        np.testing.assert_array_equal(arrays["oracle.primal"], probe.affine_oracle(bundle)["primal"])
    assert not Path(str(output) + ".tmp").exists()


def test_gpu_bootstrap_precedes_jax_import_and_cpu_selects_cpu(monkeypatch):
    events = []
    config_updates = []
    fake_jax = ModuleType("jax")
    fake_jax.__version__ = "0.4.38"
    fake_jax.config = SimpleNamespace(update=lambda key, value: config_updates.append((key, value)))
    fake_jax.devices = lambda platform=None: (SimpleNamespace(platform=platform),)

    runtime = ModuleType("scripts.hpc.orcjax_gpu_runtime")
    def initialize_gpu_backend():
        events.append("gpu-bootstrap")
        assert "jax" not in sys.modules
        sys.modules["jax"] = fake_jax
        return fake_jax, (SimpleNamespace(platform="gpu"),)
    runtime.initialize_gpu_backend = initialize_gpu_backend
    monkeypatch.setitem(sys.modules, "scripts.hpc.orcjax_gpu_runtime", runtime)
    monkeypatch.delitem(sys.modules, "jax", raising=False)
    jax, devices = probe.initialize_probe_runtime("gpu")
    assert jax is fake_jax and devices[0].platform == "gpu"
    assert events == ["gpu-bootstrap"]
    assert os.environ["JAX_PLATFORMS"] == "cuda"
    assert config_updates == [("jax_enable_x64", True), ("jax_enable_compilation_cache", False)]

    config_updates.clear()
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    jax, devices = probe.initialize_probe_runtime("cpu")
    assert jax is fake_jax and devices[0].platform == "cpu"
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert config_updates == [("jax_enable_x64", True), ("jax_enable_compilation_cache", False)]
