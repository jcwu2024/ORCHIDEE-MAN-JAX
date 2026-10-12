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
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-mock", ORCJAX_AFFINE_EXPECTED_GPU_UUID="GPU-mock",
        ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT="T7_affine_allocated_single_gpu_v1")
    env.pop("ORCJAX_NUMERICS_LAUNCHER_PYTHON", None)
    result = run(PAID, ["--runtime-child",
        "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_gpu/bin/python",
        posix(PAID.with_name("capture_shared_latent_T7_summary_operands.py"))], env)
    assert result.returncode == 2


@pytest.mark.parametrize('missing',['uuid','contract'])
def test_paid_runtime_child_requires_explicit_contract_and_uuid_before_container(missing):
    env=dict(os.environ, ORCJAX_AFFINE_EXPECTED_GPU_UUID='GPU-mock',
        ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT='T7_affine_allocated_single_gpu_v1')
    env.pop('ORCJAX_AFFINE_EXPECTED_GPU_UUID' if missing=='uuid' else 'ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT')
    result=run(PAID,['--runtime-child','must-not-execute','must-not-execute'],env)
    assert result.returncode==2


def test_paid_runtime_child_explicitly_exports_container_contract_and_uuid(tmp_path):
    target=tmp_path/'paid.sh'
    target.write_text(PAID.read_text(),newline='\n')
    shared=target.with_name('run_shared_latent_T7_batch_numerics.sh')
    shared.write_text('CODE=/unused\nmain() { printf "%s\\n" "$SINGULARITYENV_ORCJAX_AFFINE_EXPECTED_GPU_UUID" "$SINGULARITYENV_ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT"; }\n',newline='\n')
    env=dict(os.environ, ORCJAX_AFFINE_EXPECTED_GPU_UUID='GPU-mock',
        ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT='T7_affine_allocated_single_gpu_v1')
    result=run(target,['--runtime-child','unused','unused'],env)
    assert result.returncode==0 and result.stdout.decode().splitlines()==['GPU-mock','T7_affine_allocated_single_gpu_v1']


def slurm_copy(tmp_path, with_scope):
    checkout = tmp_path / "immutable checkout space"
    target = checkout / "scripts/hpc/slurm_shared_latent_T7_batch_numerics_paid.sh"
    target.parent.mkdir(parents=True)
    runtime = tmp_path / "mock runtime"
    canonical = "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-single-gpu-20261012-v1"
    target.write_text(SLURM.read_text().replace(
        "ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime", 'ROOT="' + posix(runtime) + '"'
    ).replace("CODE=" + canonical, 'CODE="' + posix(checkout) + '"'
    ).replace("#SBATCH --chdir=" + canonical, "#SBATCH --chdir=" + posix(checkout)), newline="\n")
    target.with_name("run_shared_latent_T7_batch_numerics_paid.py").write_text("# private mock; never run\n")
    diag = runtime / "outputs/diagnostics/T7-summary-affine-paid-single-gpu-20261012-v1"
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
    assert resources[14:16] == ["--nodelist", "ibc13b03n04"]
    assert resources[16:18] == ["--chdir", posix(target.parents[2])]
    assert resources[18:22] == ["--output", posix(diag / "slurm-%j.stdout.txt"), "--error", posix(diag / "slurm-%j.stderr.txt")]
    assert resources[-1].endswith("/slurm_shared_latent_T7_batch_numerics_paid.sh")
    assert command[0] == "bash" and command[1].endswith("/run_shared_latent_T7_batch_numerics_paid.sh")
    args = dict(zip(command[2:-1:2], command[3:-1:2], strict=True))
    assert set(args) == {"--checkout-root", "--operands", "--capture-receipt", "--capture-source", "--output", "--scope-json", "--ledger"}
    assert args["--output"] == posix(diag / "result.v1.json")
    assert args["--ledger"].endswith("/T7-summary-affine-paid-single-gpu-20261012-v1.ledger.json")
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


def allocation_copy(tmp_path, epochs="1699999900 1700001100\n", helper_exit=0,
                    timeout_exit=None):
    """Real copied paid/shared admission; only scheduler and launch are mocks.

    No real scheduler, process groups, GPU, or batch-environment proof here.
    The shared main computes deadlines before our guarded launch records them.
    """
    project = tmp_path / "private project"
    target = project / "scripts/hpc" / PAID.name
    target.parent.mkdir(parents=True)
    target.write_text(PAID.read_text(), newline="\n")
    helper = tmp_path / "mock cpu python.sh"
    helper_log = tmp_path / "helper.argv"
    helper_output = tmp_path / "helper.stdout"
    helper_output.write_bytes(epochs.encode())
    helper.write_text('#!/usr/bin/env bash\nprintf \'%s\\0\' '
        '"${SLURM_JOB_START_TIME-absent}" "${SLURM_JOB_END_TIME-absent}" "$@" > '
        + shlex.quote(posix(helper_log)) + '\ncat -- ' + shlex.quote(posix(helper_output))
        + f'\nexit {helper_exit}\n', newline="\n")
    helper.chmod(0o755)
    timeout_log = tmp_path / "timeout.argv"
    timeout_body = f"return {timeout_exit}" if timeout_exit is not None else 'shift; "$@"'
    shared = PAID.with_name("run_shared_latent_T7_batch_numerics.sh").read_text()
    shared = shared.replace("PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX",
        "PROJECT=" + shlex.quote(posix(project)))
    shared = shared.replace("CPU_PYTHON=$PROJECT/.venvs/orcjax_cpu/bin/python",
        "CPU_PYTHON=" + shlex.quote(posix(helper)))
    shared += '\ndate() { printf \'1700000000\\n\'; }\n'
    shared += 'timeout() { printf \'%s\\0\' "$@" > ' + shlex.quote(posix(timeout_log)) + '; ' + timeout_body + '; }\n'
    shared += 'guarded_cpu_launch() { printf \'%s\\0\' "${SLURM_JOB_START_TIME-absent}" "${SLURM_JOB_END_TIME-absent}" "$SHELL_STARTED_EPOCH" "$@"; }\n'
    target.with_name("run_shared_latent_T7_batch_numerics.sh").write_text(shared, newline="\n")
    output = project / "runtime/outputs/result.json"
    output.parent.mkdir(parents=True)
    env = dict(os.environ, SLURM_JOB_ID="123456")
    for key in ("SLURM_JOB_START_TIME", "SLURM_JOB_END_TIME",
                "ORCJAX_NUMERICS_LAUNCHER_PYTHON"):
        env.pop(key, None)
    return target, output, helper, helper_log, timeout_log, env


@pytest.mark.parametrize("original_epochs", [None, ("1699999900", "1700001100")])
def test_paid_live_scheduler_epochs_use_true_allocation_start_and_shared_deadlines(tmp_path, original_epochs):
    target, output, helper, helper_log, timeout_log, env = allocation_copy(tmp_path)
    if original_epochs is not None:
        env.update(SLURM_JOB_START_TIME=original_epochs[0], SLURM_JOB_END_TIME=original_epochs[1])
    result = run(target, ["--output", posix(output)], env)
    assert result.returncode == 0, result.stderr.decode()
    runner = posix(target.with_suffix(".py"))
    expected_helper = ["-B", runner, "--scheduler-allocation-epochs"]
    assert timeout_log.read_text().split("\0")[:-1] == ["10s", posix(helper), *expected_helper]
    assert helper_log.read_text().split("\0")[:-1] == [
        *(original_epochs or ("absent", "absent")), *expected_helper]
    actual = result.stdout.decode().split("\0")[:-1]
    # Frozen clock is 100s later than allocation start: deadline origin must
    # retain those occupied seconds, and the supervisor gets that same origin.
    assert actual == [*(original_epochs or ("absent", "absent")),
        "1699999900", "1700000910", "1700001090",
        posix(output.with_name("result.outer-timeout.json")), posix(helper),
        "-B", runner, "--shell-started-epoch", "1699999900", "--output", posix(output)]


@pytest.mark.parametrize("epochs,helper_exit,timeout_exit", [
    ("", 0, None), ("not-epoch 1700001100\n", 0, None),
    ("1699999900\n", 0, None), ("1699999900 1700001100 extra\n", 0, None),
    ("1699999900 1700001100\n1699999900 1700001100\n", 0, None),
    ("1699999900 1700001100\nignored\n", 0, None),
    ("1699999900 1700001099\n", 0, None),
    ("1699999900 1700001101\n", 0, None),
    ("1700000001 1700001201\n", 0, None),
    ("1699998800 1700000000\n", 0, None),
    ("1699999900 1700001100\n", 7, None),
    ("1699999900 1700001100\n", 0, 124),
], ids=["empty", "malformed", "missing-end", "extra-token", "multiple-valid-lines",
    "extra-line", "short-interval", "long-interval", "future", "expired", "helper-failure", "timeout"])
def test_paid_live_scheduler_failure_or_bad_epochs_never_launch_main(tmp_path, epochs, helper_exit, timeout_exit):
    target, output, helper, helper_log, timeout_log, env = allocation_copy(
        tmp_path, epochs, helper_exit, timeout_exit)
    result = run(target, ["--output", posix(output)], env)
    assert result.returncode == 2
    assert result.stdout == b""  # Mock guarded_cpu_launch was never reached.
    assert timeout_log.read_text().split("\0")[:-1] == ["10s", posix(helper),
        "-B", posix(target.with_suffix(".py")), "--scheduler-allocation-epochs"]
    assert helper_log.exists() == (timeout_exit is None)


def test_paid_live_override_rejected_before_scheduler_helper(tmp_path):
    target, output, _, helper_log, timeout_log, env = allocation_copy(tmp_path)
    env["ORCJAX_NUMERICS_LAUNCHER_PYTHON"] = "must-not-execute"
    result = run(target, ["--output", posix(output)], env)
    assert result.returncode == 2 and result.stdout == b""
    assert not helper_log.exists() and not timeout_log.exists()


@pytest.mark.parametrize('epochs,helper_exit', [('bad-clock 1700001100\n', 0), ('1699999900 1700001100\n', 7)])
def test_paid_startup_evidence_keeps_clock_reply_on_rejection_without_launch(tmp_path, epochs, helper_exit):
    target, output, _, helper_log, _, env = allocation_copy(tmp_path, epochs=epochs, helper_exit=helper_exit)
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['SLURM_JOB_GPUS'] = '3'
    env['SECRET_TOKEN'] = 'must-not-be-saved'
    result = run(target, ['--output', posix(output)], env)
    assert result.returncode == 2 and result.stdout == b''
    text = result.stderr.decode()
    assert 'PAID_STARTUP_SHELL stage=scheduler_allocation_clock_rejected exit_code=2' in text
    assert 'SLURM_JOB_ID=123456' in text and 'SLURM_JOB_GPUS=3' in text
    assert 'CUDA_VISIBLE_DEVICES=0 ORCJAX_SLURM_VISIBLE_DEVICES=0' in text
    assert epochs.rstrip().split()[0] in text and '1700001100' in text
    assert 'SECRET_TOKEN' not in text and 'must-not-be-saved' not in text
    assert helper_log.exists()


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
    assert json.loads(original) == dict(scope_id="T7-summary-affine-paid-single-gpu-20261012-v1",
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
def test_paid_runtime_required_expected_uuid_passthrough(tmp_path, runtime, present, expected):
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
    env['ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT']='T7_affine_allocated_single_gpu_v1'
    if present:
        env["ORCJAX_AFFINE_EXPECTED_GPU_UUID"] = "GPU-allocated-exact"
    argv = ["--runtime-child", "python-not-executed", "entry-not-executed"] if runtime else ["--dry-run"]
    result = run(target, argv, env)
    if runtime and not present:
        assert result.returncode == 2 and not result.stdout
        return
    assert result.returncode == 0, result.stderr.decode()
    actual = result.stdout.decode().split("\0")[:-1]
    assert actual == [expected, *argv]
