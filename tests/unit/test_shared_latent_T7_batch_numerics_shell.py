"""Execute real Bash argv expansion with CPU/container/driver mocks only."""
import os
from pathlib import Path
import shutil
import subprocess
import time
import json

import pytest


SHELL = Path(__file__).resolve().parents[2] / "scripts/hpc/run_shared_latent_T7_batch_numerics.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="Bash is required for argv verification")


def posix(path):
    path = Path(path).resolve().as_posix()
    if len(path) > 1 and path[1] == ":":
        return "/" + path[0].lower() + path[2:]
    return path


def script(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + body, encoding="utf-8", newline="\n")
    path.chmod(0o755)
    return path


def invoke(shell, args, env):
    if "ORCJAX_SHELL_TEST_BIN" in env:
        return subprocess.run([
            BASH, "-c", 'export PATH="$ORCJAX_SHELL_TEST_BIN:$PATH"; source "$1"; '
            'if [[ ${ORCJAX_SHELL_TEST_PHASE_MOCK:-} == true ]]; then start_phase_deadman() { :; }; fi; '
            'main "${@:2}"',
            "mock-exec", posix(shell), *args,
        ], env=env, capture_output=True, timeout=15)
    return subprocess.run([BASH, posix(shell), *args], env=env, capture_output=True, timeout=15)


def test_dry_run_preserves_argv_without_driver_or_container(tmp_path):
    mock = script(tmp_path / "mock python.sh", "printf '%s\\0' \"$@\"\nprintf '%s\\0' \"$JAX_PLATFORMS\" \"$CUDA_VISIBLE_DEVICES\"\n")
    forbidden = tmp_path / "forbidden"
    for name in ("readlink", "singularity", "nvidia-smi"):
        script(forbidden / name, "printf 'FORBIDDEN' >&2\nexit 99\n")
    env = dict(os.environ, ORCJAX_NUMERICS_LAUNCHER_PYTHON=posix(mock))
    env["ORCJAX_SHELL_TEST_BIN"] = posix(forbidden)
    args = ["--dry-run", "--output", "space quoted $ literal.json", "--operands", "operand [x].npz"]
    result = invoke(SHELL, args, env)
    assert result.returncode == 0, result.stderr
    received = result.stdout.decode().split("\0")[:-1]
    assert received[0] == "-B"
    assert received[1].endswith("/scripts/hpc/run_shared_latent_T7_batch_numerics.py")
    assert received[2] == "--shell-started-epoch"
    assert float(received[3]) > 0
    assert received[4:-2] == args
    assert received[-2:] == ["cpu", ""]
    assert not result.stderr


def test_live_override_rejected_before_startup(tmp_path):
    env = dict(os.environ, ORCJAX_NUMERICS_LAUNCHER_PYTHON="does-not-exist")
    result = invoke(SHELL, ["--output", "not-started.json"], env)
    assert result.returncode == 2
    assert b"requires --dry-run" in result.stderr


@pytest.mark.parametrize("args,visible", [
    (["/wrong/python", "/wrong/entry.py"], "GPU-one"),
    (["/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_gpu/bin/python", str(SHELL)], "0"),
])
def test_internal_child_rejects_noncanonical_arguments(args, visible):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible)
    env.pop("ORCJAX_NUMERICS_LAUNCHER_PYTHON", None)
    result = invoke(SHELL, ["--runtime-child", *args], env)
    assert result.returncode == 2


def test_runtime_child_container_bindings_and_uuid_are_forwarded(tmp_path):
    # A private mock tree changes only the canonical project prefix in a copy.
    # No /WORK directory, real driver, container, JAX or GPU is accessed.
    project = tmp_path / "project space"
    checkout = project / "runtime/checkouts/numerics"
    target = checkout / "scripts/hpc/run_shared_latent_T7_batch_numerics.sh"
    target.parent.mkdir(parents=True)
    target.write_text(SHELL.read_text().replace(
        "PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX",
        'PROJECT="' + posix(project) + '"'), encoding="utf-8", newline="\n")
    entry = target.with_name("probe_shared_latent_T7_batch_numerics.py")
    entry.write_text("# mock; never imported\n")
    toolchain = project / "runtime/toolchains/cuda-nvlink-12.6.85"
    toolchain.mkdir(parents=True)
    library = tmp_path / "driver lib.so"
    library.write_text("mock driver")
    bin_dir = tmp_path / "bin"
    script(bin_dir / "readlink", 'printf "%s\\n" "' + posix(library) + '"\n')
    script(bin_dir / "singularity", "printf '%s\\0' \"$SINGULARITYENV_CUDA_VISIBLE_DEVICES\" \"$SINGULARITYENV_JAX_PLATFORMS\" \"${LD_LIBRARY_PATH-unset}\" \"$@\"\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-physical-uuid", LD_LIBRARY_PATH="stale/driver")
    env["ORCJAX_NUMERICS_SHELL_STARTED_EPOCH"] = str(time.time())
    env["ORCJAX_SHELL_TEST_PHASE_MOCK"] = "true"
    env.pop("ORCJAX_NUMERICS_LAUNCHER_PYTHON", None)
    env["ORCJAX_SHELL_TEST_BIN"] = posix(bin_dir)
    python = posix(project / ".venvs/orcjax_gpu/bin/python")
    canonical_entry = posix(entry)
    result = invoke(target, ["--runtime-child", python, canonical_entry, "--output", "result space.json"], env)
    assert result.returncode == 0, result.stderr.decode()[-2300:]
    args = result.stdout.decode().split("\0")[:-1]
    assert args[:5] == ["GPU-physical-uuid", "cuda,cpu", "unset", "exec", "--nv"]
    assert posix(library) + ":/usr/lib/x86_64-linux-gnu/libcuda.so.1" in args[6]
    assert posix(library) + ":/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1" in args[6]
    assert args[-5:] == [python, "-B", canonical_entry, "--output", "result space.json"]


@pytest.mark.parametrize("entry_name,extra", [
    ("capture_shared_latent_T7_summary_operands.py", []),
    ("probe_shared_latent_T7_batch_numerics.py", ["--dry-run"]),
    ("check_daily_training_runtime.py", ["--dry-run=true"]),
])
def test_internal_entry_allowlist_and_dry_flag_are_fail_closed(entry_name, extra):
    entry = SHELL.with_name(entry_name)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-physical-uuid")
    env.pop("ORCJAX_NUMERICS_LAUNCHER_PYTHON", None)
    result = invoke(SHELL, ["--runtime-child",
        "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_gpu/bin/python",
        posix(entry), *extra], env)
    assert result.returncode == 2
    assert not result.stdout


def test_ordinary_entry_rejects_user_supplied_start_epoch():
    result = invoke(SHELL, ["--dry-run", "--shell-started-epoch", "9999999999"], dict(os.environ))
    assert result.returncode == 2


def test_outer_deadman_stops_blocked_cpu_and_preserves_marker(tmp_path):
    marker = tmp_path / "result.outer-timeout.json"
    signals = tmp_path / "signals.txt"
    # Git Bash has no setsid. Mock its PG setup and PG inspection, then translate
    # only the requested owned negative-PG signal to this actual blocking PID.
    # The blocking child ignores TERM; hard-deadline KILL is exercised for real.
    harness = r'''
source "$1"
marker=$2
signals=$3
setsid() { exec "$@"; }
ps() { printf '%s\n' "${@: -1}"; }
kill() {
  target=${@: -1}
  if [[ $target == -* ]]; then
    printf '%s\n' "$*" >> "$signals"
    builtin kill "$1" "${target#-}"
  else
    builtin kill "$@"
  fi
}
now=$(date +%s)
result=0
guarded_cpu_launch "$((now+1))" "$((now+2))" "$marker" bash -c 'trap "" TERM; exec sleep 20' || result=$?
[[ $result == 124 ]]
before=$(cat "$marker")
result=0
guarded_cpu_launch "$((now+1))" "$((now+2))" "$marker" bash -c 'exit 99' || result=$?
[[ $result == 2 && $(cat "$marker") == "$before" ]]
'''
    started = time.monotonic()
    result = subprocess.run([BASH, "-c", harness, "deadman-test", posix(SHELL),
        posix(marker), posix(signals)], capture_output=True, timeout=8)
    assert result.returncode == 0, result.stderr.decode()
    assert time.monotonic() - started < 6
    receipt = json.loads(marker.read_text())
    assert receipt["status"] == "deadline"
    assert receipt["scope_closed"] and not receipt["GPU_release_verified"]
    requested = signals.read_text().splitlines()
    assert len(requested) == 2
    assert requested[0].startswith("-TERM -- -")
    assert requested[1].startswith("-KILL -- -")
    assert requested[0].split()[-1] == requested[1].split()[-1]


@pytest.mark.parametrize("scenario", ["blocked_supervisor", "leader_exit", "caller_term"])
def test_phase_deadman_enforces_own_group_without_cpu_supervision(tmp_path, scenario):
    signals = tmp_path / "phase-signals.txt"
    # A Linux PG is unavailable in Git Bash. Map one declared fake group to
    # only its actually spawned leader, descendant and timer; preserve a
    # separate blocked CPU supervisor. Actual processes receive the signals.
    harness = r'''
source "$1"
signals=$2
scenario=$3
sleep 20 & cpu_supervisor=$!
if [[ $scenario == leader_exit ]]; then
  sleep 0.2 & phase_leader=$!
else
  bash -c 'trap "" TERM; exec sleep 20' & phase_leader=$!
fi
bash -c 'trap "" TERM; exec sleep 20' & descendant=$!
ps() { printf '%s\n' "$phase_leader"; }
kill() {
  target=${@: -1}
  if [[ $target == -* ]]; then
    [[ $target == "-$phase_leader" ]] || exit 98
    printf '%s\n' "$*" >> "$signals"
    builtin kill "$1" "$phase_leader" "$descendant" "$BASHPID" 2>/dev/null || true
  else
    builtin kill "$@"
  fi
}
now=$(date +%s)
start_phase_deadman "$phase_leader" "$((now+1))" "$((now+2))"
timer=$PHASE_TIMER_PID
if [[ $scenario == leader_exit ]]; then wait "$phase_leader"; fi
if [[ $scenario == caller_term ]]; then
  sleep 0.3
  builtin kill -TERM "$phase_leader" "$descendant" "$timer"
fi
wait "$timer" 2>/dev/null || true
wait "$phase_leader" 2>/dev/null || true
wait "$descendant" 2>/dev/null || true
builtin kill -0 "$cpu_supervisor"
builtin kill -TERM "$cpu_supervisor"
wait "$cpu_supervisor" 2>/dev/null || true
'''
    started = time.monotonic()
    result = subprocess.run([BASH, "-c", harness, "phase-deadman-test", posix(SHELL),
        posix(signals), scenario], capture_output=True, timeout=8)
    assert result.returncode == 0, result.stderr.decode()
    assert time.monotonic() - started < 6
    requested = signals.read_text().splitlines()
    assert requested[-1].startswith("-KILL -- -")
    if scenario != "leader_exit":
        assert requested[0].startswith("-TERM -- -")
        assert requested[0].split()[-1] == requested[-1].split()[-1]


def test_phase_deadman_rejects_non_session_leader():
    harness = 'source "$1"; ps() { printf "999999\\n"; }; start_phase_deadman "$$" 1 2'
    result = subprocess.run([BASH, "-c", harness, "phase-reject-test", posix(SHELL)],
        capture_output=True, timeout=3)
    assert result.returncode == 2
