"""Focused paid Bash admission/argv tests; all writable paths are tmp_path."""
import os
import json
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
PAID = ROOT / "scripts/hpc/run_shared_latent_T7_batch_numerics_paid.sh"
SLURM = PAID.with_name("slurm_shared_latent_T7_batch_numerics_paid.sh")
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="Bash required")


def posix(path):
    value = Path(path).resolve().as_posix()
    return "/" + value[0].lower() + value[2:] if value[1:2] == ":" else value


def run(shell, args, env=None):
    return subprocess.run([BASH, posix(shell), *args], env=env or os.environ,
        capture_output=True, timeout=10)


def test_paid_dry_run_saves_slurm_visibility_before_cpu_clear(tmp_path):
    child = tmp_path / "mock python.sh"
    child.write_text('#!/usr/bin/env bash\nprintf \'%s\\0\' "$ORCJAX_SLURM_VISIBLE_DEVICES" "$CUDA_VISIBLE_DEVICES" "$JAX_PLATFORMS" "$@"\n', newline="\n")
    child.chmod(0o755)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="2", SLURM_JOB_ID="mock-job",
        ORCJAX_NUMERICS_LAUNCHER_PYTHON=posix(child))
    argv = ["--dry-run", "--output", "quoted result $ literal.json", "--scope-json", "scope space.json"]
    result = run(PAID, argv, env)
    assert result.returncode == 0, result.stderr.decode()
    actual = result.stdout.decode().split("\0")[:-1]
    assert actual[:4] == ["2", "", "cpu", "-B"]
    assert actual[4].endswith("/scripts/hpc/run_shared_latent_T7_batch_numerics_paid.py")
    assert actual[5] == "--shell-started-epoch" and float(actual[6]) > 0
    assert actual[7:] == argv


@pytest.mark.parametrize("args", [["--output", "no-launch.json"],
    ["--dry-run", "--shell-started-epoch", "9999999999"]])
def test_paid_entry_rejects_live_override_and_epoch_spoof(args):
    env = dict(os.environ, ORCJAX_NUMERICS_LAUNCHER_PYTHON="must-not-execute")
    result = run(PAID, args, env)
    assert result.returncode == 2


def test_paid_runtime_child_reuses_strict_free_allowlist():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-mock")
    env.pop("ORCJAX_NUMERICS_LAUNCHER_PYTHON", None)
    result = run(PAID, ["--runtime-child",
        "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_gpu/bin/python",
        posix(PAID.with_name("capture_shared_latent_T7_summary_operands.py"))], env)
    assert result.returncode == 2


def slurm_copy(tmp_path, with_scope):
    checkout = tmp_path / "immutable checkout space"
    target = checkout / "scripts/hpc/slurm_shared_latent_T7_batch_numerics_paid.sh"
    target.parent.mkdir(parents=True)
    runtime = tmp_path / "mock runtime"
    canonical = "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-20261011-v1"
    target.write_text(SLURM.read_text().replace(
        "ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime", 'ROOT="' + posix(runtime) + '"'
    ).replace("CODE=" + canonical, 'CODE="' + posix(checkout) + '"'
    ).replace("#SBATCH --chdir=" + canonical, "#SBATCH --chdir=" + posix(checkout)), newline="\n")
    target.with_name("run_shared_latent_T7_batch_numerics_paid.py").write_text("# private mock; never run\n")
    diag = runtime / "outputs/diagnostics/T7-summary-affine-paid-gpu-20261011-v1"
    diag.mkdir(parents=True)
    if with_scope:
        (diag / "scope.v1.json").write_text("{}\n")
    return target, diag, runtime


def test_slurm_dry_run_prints_exact_paid_resource_and_downstream_argv(tmp_path):
    target, diag, runtime = slurm_copy(tmp_path, with_scope=True)
    result = run(target, ["--dry-run"])
    assert result.returncode == 0, result.stderr.decode()
    resources, command = map(shlex.split, result.stdout.decode().splitlines())
    assert resources[:14] == ["sbatch", "--partition", "gnall", "--nodes", "1", "--ntasks", "1",
        "--cpus-per-task", "4", "--gres", "gpu:1", "--time", "00:20:00", "--no-requeue"]
    assert resources[14:16] == ["--nodelist", "ibc13b03n01"]
    assert resources[16:18] == ["--chdir", posix(target.parents[2])]
    assert resources[18:22] == ["--output", posix(diag / "slurm-%j.stdout.txt"), "--error", posix(diag / "slurm-%j.stderr.txt")]
    assert resources[-1].endswith("/slurm_shared_latent_T7_batch_numerics_paid.sh")
    assert command[0] == "bash" and command[1].endswith("/run_shared_latent_T7_batch_numerics_paid.sh")
    args = dict(zip(command[2:-1:2], command[3:-1:2], strict=True))
    assert set(args) == {"--checkout-root", "--operands", "--capture-receipt", "--capture-source", "--output", "--scope-json", "--ledger"}
    assert args["--output"] == posix(diag / "result.v1.json")
    assert args["--ledger"].endswith("/T7-summary-affine-paid-gpu-20261011-v1.ledger.json")
    assert args["--operands"] == posix(runtime / "transfer/T7-summary-recovery-20261011-v1/summary-recovery.v1.operands.npz")
    assert command[-1] == "--dry-run"
    original = SLURM.read_text()
    assert "#SBATCH --exclusive" not in original
    assert "--hint" not in original
    assert "/slurm-%j.stdout.txt" in original and "/slurm-%j.stderr.txt" in original


def test_slurm_rejects_missing_prepared_scope_without_writes(tmp_path):
    target, diag, _ = slurm_copy(tmp_path, with_scope=False)
    result = run(target, ["--dry-run"])
    assert result.returncode == 2 and list(diag.iterdir()) == []


def test_slurm_live_cannot_run_outside_allocation(tmp_path):
    target, _, _ = slurm_copy(tmp_path, with_scope=True)
    env = dict(os.environ)
    env.pop("SLURM_JOB_ID", None)
    result = run(target, [], env)
    assert result.returncode == 2


@pytest.mark.parametrize("start,end", [("", ""), ("not-epoch", "9999999999"),
    ("9999999999", "10000000000"), ("1", "1202")])
def test_paid_live_rejects_missing_or_invalid_allocation_epochs(start, end):
    env = dict(os.environ, SLURM_JOB_ID="mock-only", SLURM_JOB_START_TIME=start,
        SLURM_JOB_END_TIME=end)
    env.pop("ORCJAX_NUMERICS_LAUNCHER_PYTHON", None)
    result = run(PAID, ["--output", "never-started.json"], env)
    assert result.returncode == 2


def test_slurm_start_consumed_before_cpu_admission_and_cannot_retry(tmp_path):
    target, diag, _ = slurm_copy(tmp_path, with_scope=True)
    child = target.with_name("run_shared_latent_T7_batch_numerics_paid.sh")
    calls = tmp_path / "mock-child-calls.txt"
    child.write_text('#!/usr/bin/env bash\nprintf \'child-before-python\\n\' >> "' +
        posix(calls) + '"\nexit 7\n', newline="\n")
    env = dict(os.environ, SLURM_JOB_ID="123456")
    result = run(target, [], env)
    assert result.returncode == 7
    marker = diag / "slurm-attempt.v1.json"
    original = marker.read_bytes()
    assert json.loads(original) == dict(scope_id="T7-summary-affine-paid-gpu-20261011-v1",
        job_id="123456", status="scheduler_started", retry_allowed=False)
    assert calls.read_text().splitlines() == ["child-before-python"]
    second = run(target, [], env)
    assert second.returncode == 2 and marker.read_bytes() == original
    assert calls.read_text().splitlines() == ["child-before-python"]
    dry = run(target, ["--dry-run"], env)
    assert dry.returncode == 0 and marker.read_bytes() == original


def test_slurm_rejects_unsafe_jobid_without_start_marker(tmp_path):
    target, diag, _ = slurm_copy(tmp_path, with_scope=True)
    result = run(target, [], dict(os.environ, SLURM_JOB_ID='bad"job-id'))
    assert result.returncode == 2
    assert not (diag / "slurm-attempt.v1.json").exists()


def test_slurm_spooled_copy_keeps_original_frozen_checkout(tmp_path):
    original, _, _ = slurm_copy(tmp_path, with_scope=True)
    spool = tmp_path / "scheduler spool" / "job123" / "slurm_script"
    spool.parent.mkdir(parents=True)
    spool.write_text(original.read_text(), newline="\n")
    result = run(spool, ["--dry-run"])
    assert result.returncode == 0, result.stderr.decode()
    resources, command = map(shlex.split, result.stdout.decode().splitlines())
    checkout = posix(original.parents[2])
    assert resources[resources.index("--chdir") + 1] == checkout
    assert command[1] == checkout + "/scripts/hpc/run_shared_latent_T7_batch_numerics_paid.sh"
    assert command[command.index("--checkout-root") + 1] == checkout
    assert "scheduler spool" not in command[1]


@pytest.mark.parametrize("missing", ["scope", "code"])
def test_slurm_early_rejection_already_consumed_before_scope_or_code_gate(tmp_path, missing):
    target, diag, _ = slurm_copy(tmp_path, with_scope=missing != "scope")
    if missing == "code":
        target.write_text(target.read_text().replace(
            'CODE="' + posix(target.parents[2]) + '"',
            'CODE="' + posix(tmp_path / "missing checkout") + '"'), newline="\n")
    env = dict(os.environ, SLURM_JOB_ID="234567")
    first = run(target, [], env)
    marker = diag / "slurm-attempt.v1.json"
    assert first.returncode == 2 and marker.exists()
    original = marker.read_bytes()
    assert json.loads(original)["retry_allowed"] is False
    second = run(target, [], env)
    assert second.returncode == 2 and marker.read_bytes() == original
    assert b"no automatic retry" in second.stderr


@pytest.mark.parametrize("unavailable", ["absent_directory", "non_directory_destination"])
def test_slurm_marker_write_failure_is_unknown_not_retryable(tmp_path, unavailable):
    target, diag, _ = slurm_copy(tmp_path, with_scope=True)
    unavailable_dir = tmp_path / "unavailable output"
    if unavailable == "non_directory_destination":
        unavailable_dir.write_text("private regular file; cannot hold a marker\n")
    target.write_text(target.read_text().replace("DIAG=$ROOT/outputs/diagnostics/$ID",
        'DIAG="' + posix(unavailable_dir) + '"'), newline="\n")
    result = run(target, [], dict(os.environ, SLURM_JOB_ID="345678"))
    assert result.returncode == 2
    assert b"state unknown" in result.stderr and b"no automatic retry" in result.stderr
    assert b"pre-submission receipt" in result.stderr
    assert not (unavailable_dir / "slurm-attempt.v1.json").exists()
    assert unavailable_dir.exists() == (unavailable == "non_directory_destination")


@pytest.mark.parametrize("runtime,present,expected", [
    (True, True, "GPU-allocated-exact"), (True, False, "unset"),
    (False, True, "unset"),
])
def test_paid_runtime_optional_expected_uuid_passthrough(tmp_path, runtime, present, expected):
    # Mock only the source helper's main. The real paid source is copied into
    # tmp_path and executes its environment branch; no driver/runtime is entered.
    target = tmp_path / "private paid helper" / PAID.name
    target.parent.mkdir(parents=True)
    target.write_text(PAID.read_text(), newline="\n")
    target.with_name("run_shared_latent_T7_batch_numerics.sh").write_text(
        'CODE=/mock-checkout\nmain() { printf \'%s\\0\' "${SINGULARITYENV_ORCJAX_AFFINE_EXPECTED_GPU_UUID-unset}" "$@"; }\n', newline="\n")
    env = dict(os.environ)
    env.pop("SINGULARITYENV_ORCJAX_AFFINE_EXPECTED_GPU_UUID", None)
    env.pop("ORCJAX_AFFINE_EXPECTED_GPU_UUID", None)
    if present:
        env["ORCJAX_AFFINE_EXPECTED_GPU_UUID"] = "GPU-allocated-exact"
    argv = ["--runtime-child", "python-not-executed", "entry-not-executed"] if runtime else ["--dry-run"]
    result = run(target, argv, env)
    assert result.returncode == 0, result.stderr.decode()
    actual = result.stdout.decode().split("\0")[:-1]
    assert actual == [expected, *argv]
