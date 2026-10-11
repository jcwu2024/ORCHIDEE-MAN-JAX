import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.hpc import global_gpu_lifecycle_coordinator as coordinator

CODE_SHA = "1fdacde59d86f696f384b56d69292c8f6fa8d30b"
GPU_UUID = "GPU-test-v100"
GPU_TOTAL = 32 * 1024**3


@pytest.fixture(autouse=True)
def canonical_entry_environment(monkeypatch):
    monkeypatch.setenv("ORCHJAX_GLOBAL_BATCH_ENTRY", "benchmark_global_gpu_lifecycle.py")
    monkeypatch.setenv("SINGULARITYENV_ORCHJAX_GLOBAL_BATCH_ENTRY", "benchmark_global_gpu_lifecycle.py")
    monkeypatch.setenv("ORCHJAX_PHYSICAL_GPU_UUID", GPU_UUID)
    monkeypatch.setenv("ORCHJAX_GPU_WINDOW_START_EPOCH", str(time.time()))


def child_template(code_dir=Path("/code")):
    code_dir = str(code_dir).replace("\\", "/")
    return [
        "/rmprog/slurm/v22.05.7/bin/srun", "--ntasks=1", "--cpus-per-task=4", "--cpu-bind=cores",
        "/usr/bin/time", "-v", "singularity", "exec", "--nv", "--bind",
        "/WORK:/WORK,/usr/lib64/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "--pwd", code_dir,
        "/apps/soft/sif/foundationpose", "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_gpu/bin/python",
        f"{code_dir}/scripts/hpc/global_batch_gpu_entry.py",
        "--prepared-input", "/frozen/prepared.pkl", "--full-reference-dir", "/frozen/reference",
        "--memory-baseline-dir", "/frozen/memory", "--b1024-evidence-dir", "/frozen/b1024",
        "--config", f"{code_dir}/config.yaml", "--asset-root", "/frozen/assets", "--output-dir", "__ARM_OUTPUT_DIR__",
        "--warm-repeats", "5", "--budget-seconds", "7200", "--stage-shape", "1024",
        "--measurement-strategy", "__MEASUREMENT_STRATEGY__", "--expected-code-sha", CODE_SHA,
    ]


def identity(strategy, *, sampling_order=coordinator.SAMPLING_ORDER, code_sha=CODE_SHA,
             uuid=GPU_UUID, total=GPU_TOTAL, runtime="runtime-v1"):
    result = {
        "contract": coordinator.CONTRACT,
        "process_strategy": coordinator.PROCESS_STRATEGY,
        "stage_name": coordinator.STAGE_NAME,
        "stage_shape": 1024,
        "total_lanes": 2048,
        "warm_repeats": 5,
        "budget_seconds": 7200,
        "candidate_shapes": [1024],
        "measurement_strategy": strategy,
        "sampling_order": sampling_order,
        "wrapper_resident_input_policy": coordinator.POLICIES[strategy],
        "expected_code_sha": code_sha,
        "science_sha256": "8c8b1dfd0b207e96bbb81c2d3eb85ae6edccb9ed3c6b77fb470dabc58c5fa77d",
        "science_source_pin": CODE_SHA,
        "comparison_reference": {"old_science_sha256": "old-science", "sha256": "frozen-full64"},
        "runtime": {"jax": runtime, "jaxlib": runtime, "device_details": "V100", "x64_enabled": True},
        "precision_policy": {"jax_enable_x64": True},
        "gate": {"atol": 1e-8, "rtol": 1e-10, "discrete_schema_dtype": "exact"},
        "workload": {"lane_days": 28672},
        "run_binding": {
            "physical_gpu": {"uuid": uuid, "total_bytes": total},
            "Slurm_job_id": "job-test",
            "hostname": "node-test",
            "server_resources": {"allocated_CPU_request_count": 4, "CUDA_VISIBLE_DEVICES_preserved": "0",
                                 "hostname": "node-test", "Slurm": {"SLURM_JOB_ID": "job-test",
                                 "SLURM_JOB_PARTITION": "gnall", "SLURM_CPUS_PER_TASK": "4",
                                 "SLURM_NTASKS": "1"}},
            "allocator": "default",
            "preallocate": False,
            "window_start_epoch": 1000.0,
        },
    }
    return result


def fixture(tmp_path, *, validator_mutation=None):
    output = tmp_path / "run"
    reference = tmp_path / "reference"
    reference.mkdir()
    args = SimpleNamespace(output_dir=output, expected_code_sha=CODE_SHA, full_reference_dir=Path("/frozen/reference"),
                           budget_seconds=7200, dry_run=False, gpu_identity=tmp_path / "gpu.json",
                           code_dir=coordinator.ROOT, child_command=child_template(coordinator.ROOT))
    args.gpu_identity.write_text(json.dumps({"status": "verified_metadata_only", "uuid": GPU_UUID,
                                             "name": "Tesla V100-SXM2-32GB"}), encoding="utf-8")
    calls = []
    origin_start = {"value": time.time()}

    def run_child(strategy, command, remaining):
        calls.append(strategy)
        stage_dir = output / strategy / "capacity" / "B1024"
        stage_dir.mkdir(parents=True)
        (stage_dir / "sealed-stage.pkl").write_bytes(b"sealed " + strategy.encode())
        (stage_dir / "array_acceptance.json").write_text('{"passed": true}\n', encoding="utf-8")
        origin_start["value"] = time.time()
        return {"returncode": 0, "command_process_pid": 4100 + len(calls)}

    def validate(directory, *, strategy, expected_code_sha, common_identity=None, reference_dir=None, acceptance_path=None):
        assert expected_code_sha == CODE_SHA
        assert reference_dir == Path("/frozen/reference")
        stage = directory / "sealed-stage.pkl"
        acceptance = directory / "array_acceptance.json"
        proof = identity(strategy)
        if validator_mutation:
            validator_mutation(strategy, proof)
        common = {key: value for key, value in proof.items()
                  if key not in ("measurement_strategy", "wrapper_resident_input_policy")}
        if common_identity is not None and common_identity != common:
            raise ValueError("common identity mismatch")
        core_rate = 100. if strategy == "phased_resident_control" else 120.
        wrapper_rate = 80. if strategy == "phased_resident_control" else 100.
        return {
            "identity": proof,
            "common_identity": common,
            "stage": {"status": "complete", "gates": [{"passed": True}], "timing": {
                "core": {"lane_days": 28672, "median_seconds": 28672 / core_rate, "lane_days_per_second": core_rate,
                         "samples_seconds": [28672 / core_rate] * 5},
                "wrapper": {"lane_days": 28672, "median_seconds": 28672 / wrapper_rate, "lane_days_per_second": wrapper_rate,
                            "samples_seconds": [28672 / wrapper_rate] * 5}}},
            "path": str(stage.resolve()),
            "sha256": coordinator.digest_file(stage)[0],
            "bytes": stage.stat().st_size,
            "execution_origin": {"pid": 5000 + len(calls), "process_start_epoch": origin_start["value"]},
            "array_acceptance_path": str(acceptance.resolve()),
            "array_acceptance_sha256": coordinator.digest_file(acceptance)[0],
        }

    snapshots = []

    def snapshot(uuid):
        snapshots.append(uuid)
        return {"uuid": GPU_UUID, "total_bytes": GPU_TOTAL, "free_bytes": 28 * 1024**3}

    return args, calls, validate, run_child, snapshot


def test_dry_run_expands_two_b1024_arms_without_creating_or_running(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path / "new-run", expected_code_sha=CODE_SHA,
                           full_reference_dir=Path("/frozen/reference"), budget_seconds=7200,
                           dry_run=True, gpu_identity=None, code_dir=Path("/code"), child_command=child_template())
    report = coordinator.coordinate(args)
    assert report["status"] == "dry_run"
    assert report["strategies"] == ["phased_resident_control", "phased_release"]
    assert report["JAX_imported"] is False
    assert report["model_execution"] is False
    assert report["array_validation"] is False
    assert report["commands"]["phased_resident_control"][-4:] == [
        "--measurement-strategy", "phased_resident_control", "--expected-code-sha", CODE_SHA,
    ]
    assert report["commands"]["phased_release"][
        report["commands"]["phased_release"].index("--output-dir") + 1
    ].replace("\\", "/").endswith("/phased_release/capacity/B1024")
    assert not args.output_dir.exists()


def test_windows_cli_dry_run_preserves_canonical_posix_work_output_paths():
    args = SimpleNamespace(output_dir="/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/gpu-lifecycle-dry-run",
                           expected_code_sha=CODE_SHA, full_reference_dir=Path("/frozen/reference"),
                           budget_seconds=7200, dry_run=True, gpu_identity=None, code_dir=Path("/code"),
                           child_command=child_template())
    report = coordinator.coordinate(args)
    for strategy in coordinator.STRATEGIES:
        output_dir = report["commands"][strategy][report["commands"][strategy].index("--output-dir") + 1]
        assert output_dir == f"/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/gpu-lifecycle-dry-run/{strategy}/capacity/B1024"
    evidence_root = os.environ.get("ORCHJAX_LIFECYCLE_PREPARATION_EVIDENCE")
    if evidence_root:
        evidence_path = Path(evidence_root) / "coordinator-dry-run-canonical-work.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_two_arms_require_full_common_identity_and_independent_acceptance(tmp_path):
    args, calls, validate, run_child, snapshot = fixture(tmp_path)
    report = coordinator.coordinate(args, run_child=run_child, validate_receipt=validate, snapshot=snapshot)
    assert report["status"] == "completed_comparison", report.get("comparison_error") or report
    assert report["comparison_allowed"] is True
    assert report["ratio_published"] is True
    assert report["comparison"]["boundaries"]["core"]["release_over_control_throughput_ratio"] == pytest.approx(1.2)
    assert report["comparison"]["boundaries"]["wrapper"]["release_over_control_throughput_ratio"] == pytest.approx(1.25)
    assert (args.output_dir / "comparison.json").is_file()
    assert calls == ["phased_resident_control", "phased_release"]
    for strategy in coordinator.STRATEGIES:
        arm = report["children"][strategy]
        assert arm["status"] == "accepted"
        assert arm["sealed_receipt"]["array_acceptance_sha256"]
        assert arm["common_identity"]["sampling_order"] == coordinator.SAMPLING_ORDER
    assert (args.output_dir / "orchestration.json").is_file()
    assert (args.output_dir / "orchestration-events.jsonl").is_file()


@pytest.mark.parametrize("mutation", [
    lambda strategy, proof: proof.update(sampling_order="interleaved_core_wrapper"),
    lambda strategy, proof: proof.update(expected_code_sha="0" * 40),
    lambda strategy, proof: proof["run_binding"].update(allocator="platform"),
])
def test_first_arm_identity_failure_prevents_second_dispatch(tmp_path, mutation):
    args, calls, validate, run_child, snapshot = fixture(tmp_path, validator_mutation=mutation)
    report = coordinator.coordinate(args, run_child=run_child, validate_receipt=validate, snapshot=snapshot)
    assert report["status"] == "first_arm_failed"
    assert calls == ["phased_resident_control"]
    assert report["ratio_published"] is False
    assert (args.output_dir / "phased_resident_control/capacity/B1024/sealed-stage.pkl").is_file()


def test_second_arm_sampling_order_or_runtime_mismatch_stops_without_comparison(tmp_path):
    def mutate(strategy, proof):
        if strategy == "phased_release":
            proof["sampling_order"] = "interleaved_core_wrapper"

    args, calls, validate, run_child, snapshot = fixture(tmp_path, validator_mutation=mutate)
    report = coordinator.coordinate(args, run_child=run_child, validate_receipt=validate, snapshot=snapshot)
    assert calls == list(coordinator.STRATEGIES)
    assert report["status"] == "second_arm_failed"
    assert report["comparison_allowed"] is False
    assert report["ratio_published"] is False
    assert report["children"]["phased_resident_control"]["status"] == "accepted"


def test_nonzero_first_arm_preserves_failure_and_never_validates_or_dispatches_second(tmp_path):
    args, calls, validate, _, snapshot = fixture(tmp_path)
    def fail_first(strategy, command, remaining):
        calls.append(strategy)
        return {"returncode": 17, "command_process_pid": 4100}
    report = coordinator.coordinate(args, run_child=fail_first, validate_receipt=validate, snapshot=snapshot)
    assert report["status"] == "first_arm_failed"
    assert calls == ["phased_resident_control"]
    assert report["children"]["phased_resident_control"]["returncode"] == 17


def test_delayed_second_arm_journal_write_rechecks_3600_second_guard(tmp_path, monkeypatch):
    args, calls, validate, _, snapshot = fixture(tmp_path)
    now = [time.time()]
    monkeypatch.setenv("ORCHJAX_GPU_WINDOW_START_EPOCH", str(now[0]))
    original_flush = coordinator.Journal.flush

    def delayed_flush(journal, event, **details):
        original_flush(journal, event, **details)
        if event == "child_dispatch_started" and details.get("strategy") == "phased_release":
            now[0] += 101

    monkeypatch.setattr(coordinator.Journal, "flush", delayed_flush)

    def fake_clock():
        return now[0]

    def run_child(strategy, command, remaining):
        calls.append(strategy)
        start = now[0]
        if strategy == "phased_resident_control":
            now[0] += 3500
        stage_dir = args.output_dir / strategy / "capacity" / "B1024"
        stage_dir.mkdir(parents=True)
        (stage_dir / "sealed-stage.pkl").write_bytes(b"sealed " + strategy.encode())
        (stage_dir / "array_acceptance.json").write_text('{"passed": true}\n', encoding="utf-8")
        return {"returncode": 0, "command_process_pid": 4100 + len(calls),
                "worker_start_epoch": start}

    def validate_with_fake_origin(directory, **kwargs):
        receipt = validate(directory, **kwargs)
        receipt["execution_origin"]["process_start_epoch"] = now[0] if kwargs["strategy"] == "phased_release" else now[0] - 3500
        return receipt

    report = coordinator.coordinate(args, run_child=run_child, validate_receipt=validate_with_fake_origin,
                                    snapshot=snapshot, clock=fake_clock)
    assert report["status"] == "second_arm_failed"
    assert calls == ["phased_resident_control"]
    assert report["children"]["phased_release"]["error"] == "allocation time guard failed immediately before child dispatch"
    assert report["children"]["phased_release"]["status"] == "failed"
    assert report["ratio_published"] is False


def test_child_command_rejects_b2048_and_wrong_sampling_contract():
    command = child_template()
    command[command.index("1024")] = "2048"
    with pytest.raises(ValueError, match="B1024 only"):
        coordinator.validate_command_template(command, CODE_SHA, code_dir=Path("/code"),
                                              full_reference_dir=Path("/frozen/reference"), dry_run=True)
    command = child_template()
    command[command.index("--measurement-strategy") + 1] = "interleaved"
    with pytest.raises(ValueError, match="measurement strategy"):
        coordinator.validate_command_template(command, CODE_SHA, code_dir=Path("/code"),
                                              full_reference_dir=Path("/frozen/reference"), dry_run=True)
