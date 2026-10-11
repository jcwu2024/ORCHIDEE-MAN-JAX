"""Read-only target Bash and local non-model CLI review of the fixed execution pin."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[4]
    assert Path.cwd().resolve() == repo
    out = Path(__file__).resolve().parent
    pin = json.loads((out / "execution-code-pin.json").read_text(encoding="utf-8"))
    sha, base = pin["execution_code_commit"], pin["base_execution_commit"]
    source = pin["source_research_commit"]

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo)

    branch = git("branch", "--show-current").decode().strip()
    head = git("rev-parse", "HEAD").decode().strip()
    dirty = git("status", "--porcelain", "--untracked-files=all").decode().strip()
    assert head == source and branch == "research/global-throughput" and not dirty
    assert git("rev-parse", sha + "^").decode().strip() == base
    assert not git("diff", "--name-only", base, sha, "--", "jax_orchidee").strip()
    assert git("rev-parse", sha + ":jax_orchidee").decode().strip() == pin["scientific_source"]["science_tree_oid"]
    assert git("rev-parse", "HEAD:jax_orchidee").decode().strip() == pin["scientific_source"]["research_science_tree_oid"]
    assert set(git("diff", "--name-only", base, sha).decode().splitlines()) == set(pin["harness_paths"])
    blob = git("show", sha + ":scripts/hpc/slurm_global_server_cpu_48core.sh")
    assert hashlib.sha256(blob).hexdigest() == pin["harness_blob_sha256"]["scripts/hpc/slurm_global_server_cpu_48core.sh"]
    assert b"\r" not in blob
    target = shlex.join([
        "env", "EXPECTED_CODE_COMMIT=" + sha, "CODE_DIR=" + pin["target_checkout"],
        "bash", "-s", "--", "--dry-run",
    ])
    ssh = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
        "-o", "HostName=192.168.11.2", "cln01", target,
    ]
    remote = subprocess.run(ssh, input=blob, capture_output=True, timeout=35)
    receipt = {
        "schema": "cpu48-b64-pinned-interface-review-v1",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "execution_code_commit": sha,
        "source_research_commit": source,
        "base_execution_commit": base,
        "exact_scientific_subtree_unchanged": True,
        "research_physical_subtree_preserved": True,
        "declared_thirteen_harness_test_paths_only": True,
        "target_bash": {
            "ssh_argv": ssh,
            "stdin_source": sha + ":scripts/hpc/slurm_global_server_cpu_48core.sh",
            "stdin_sha256": hashlib.sha256(blob).hexdigest(),
            "exit_code": remote.returncode,
            "stdout": remote.stdout.decode("utf-8", errors="replace"),
            "stderr": remote.stderr.decode("utf-8", errors="replace"),
            "remote_writes": False,
            "remote_Python_JAX_model_execution": False,
        },
    }
    assert remote.returncode == 0, receipt["target_bash"]
    lines = receipt["target_bash"]["stdout"].splitlines()
    commands = [line for line in lines if line.startswith("coordinator=")]
    assert len(commands) == 1
    argv = shlex.split(commands[0][len("coordinator="):])
    assert argv[:2] == [
        "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python",
        pin["target_checkout"] + "/scripts/hpc/global_cpu48_b64_coordinator.py",
    ]
    arguments = argv[2:]
    assert arguments[arguments.index("--expected-code-commit") + 1] == sha
    assert arguments[arguments.index("--code-dir") + 1] == pin["target_checkout"]
    assert arguments[-1] == "--dry-run"
    local_arguments = arguments[:]
    local_arguments[local_arguments.index("--python") + 1] = sys.executable
    local_arguments[local_arguments.index("--entry") + 1] = str(repo / "scripts/dev/benchmark_global_server_cpu.py")
    local_command = [
        sys.executable, "-B", str(repo / "scripts/hpc/global_cpu48_b64_coordinator.py"),
        *local_arguments,
    ]
    local_env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "ORCHJAX_PARENT_CPU_COUNT": "48"}
    local_output = Path(local_arguments[local_arguments.index("--output-dir") + 1])
    output_existed = local_output.exists()
    local = subprocess.run(local_command, cwd=repo, env=local_env, capture_output=True, text=True, timeout=35)
    assert local.returncode == 0, local.stderr
    dry = json.loads(local.stdout)
    assert dry["status"] == "dry_run" and dry["expected_execution_commit"] == sha
    assert dry["model_execution"] is False and dry["JAX_imported"] is False
    assert dry["allocation"]["parent_cpu_count"] == 48 and dry["allocation"]["wall_seconds"] == 5400
    for mode, cpus in [("scalar", 1), ("batch", 48)]:
        actual = dry["runner_cli_dry_runs"][mode]
        observed = actual["runner_result"]
        assert observed["model_execution"] is False and observed["JAX_imported"] is False
        assert observed["CPU_budget"] == cpus and observed["warm_repeats"] == 5 and observed["sizes"] == [64]
        assert observed["parent_CPU_count_declared"] == "48"
        assert actual["command"].count("--cpu-reference-dir") == (1 if mode == "batch" else 0)
    assert local_output.exists() == output_existed
    assert not git("status", "--porcelain", "--untracked-files=all").strip()
    receipt["local_coordinator_to_runner_cli"] = {
        "status": "passed",
        "command": local_command,
        "adaptation": (
            "Only Python and benchmark entry paths substituted for local actual CLI validation; "
            "execution pin, target code-dir, input/reference and run-root bindings retained. "
            "This is not target Python runtime validation."
        ),
        "exit_code": local.returncode,
        "result": dry,
        "stderr": local.stderr,
        "output_directory_existence_changed": False,
    }
    receipt["working_tree_after"] = {"HEAD": head, "branch": branch, "dirty": False}
    receipt.update({
        "bundle_transferred": False, "deployment": False, "submission": False,
        "scientific_acceptance": False, "performance_acceptance": False,
    })
    destination = out / "pinned-interface-review.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "passed", "execution_code_commit": sha, "target_Bash": "passed",
        "actual_local_coordinator_and_scalar_batch_runner_CLI": "passed",
        "receipt": str(destination), "research_checkout_clean": True,
    }))


if __name__ == "__main__":
    main()
