import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CODE_SHA = "1fdacde59d86f696f384b56d69292c8f6fa8d30b"


def fixture(tmp_path):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("Git Bash is required for target-shell argument expansion")
    project = tmp_path / "project"
    code = project / "runtime/worktrees/declared-code"
    scripts = code / "scripts/hpc"
    scripts.mkdir(parents=True)
    source = (ROOT / "scripts/hpc/slurm_global_gpu_lifecycle.sh").read_text(encoding="utf-8")
    source = source.replace("PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX", f"PROJECT={shlex.quote(project.as_posix())}")
    launcher = scripts / "slurm_global_gpu_lifecycle.sh"
    launcher.write_text(source, encoding="utf-8")
    shutil.copyfile(ROOT / "scripts/hpc/global_gpu_lifecycle_coordinator.py",
                    scripts / "global_gpu_lifecycle_coordinator.py")

    git_log = code / "git-invocations.txt"
    env = os.environ.copy()
    for key in ("SLURM_JOB_ID", "SLURM_JOB_NUM_NODES", "SLURM_NNODES", "SLURM_NTASKS", "SLURM_CPUS_PER_TASK",
                "SLURM_JOB_PARTITION", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES", "RUN_ROOT",
                "XLA_PYTHON_CLIENT_ALLOCATOR", "SINGULARITYENV_XLA_PYTHON_CLIENT_ALLOCATOR"):
        env.pop(key, None)
    env.update({
        "CODE_DIR": code.as_posix(),
        "EXPECTED_CODE_SHA": CODE_SHA,
        "MSYS_NO_PATHCONV": "1",
    })
    return bash, project, code, launcher, git_log, env


@pytest.mark.parametrize("spooled", [False, True])
def test_launcher_gitbash_dry_run_and_real_coordinator_expansion(tmp_path, spooled):
    bash, project, code, launcher, git_log, env = fixture(tmp_path)
    if spooled:
        spool = tmp_path / "slurm-spool/job-not-submitted"
        spool.mkdir(parents=True)
        launcher = spool / "slurm_script"
        shutil.copyfile(project / "runtime/worktrees/declared-code/scripts/hpc/slurm_global_gpu_lifecycle.sh", launcher)

    git_shim = (
        "git() { printf '%s\\t%s\\n' \"$PWD\" \"$*\" >> \"$CODE_DIR/git-invocations.txt\"; "
        "[[ ${1:-} != -C ]] || return 91; "
        f"if [[ ${{1:-}} == rev-parse && ${{2:-}} == HEAD ]]; then printf '%s\\n' {CODE_SHA}; "
        "elif [[ ${1:-} == status ]]; then return 0; else return 92; fi; }; "
        "source \"$1\" --dry-run"
    )
    result = subprocess.run([bash, "-c", git_shim, "lifecycle-test", launcher.as_posix()],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "partition=gnall nodes=1 tasks=1 cpus=4 gpus=1 wall=02:00:00" in result.stdout
    assert "no_requeue=true exclusive=false" in result.stdout
    assert "cost_primary_cny=4.80 cost_conservative_cny=4.96" in result.stdout
    assert "selected_node=requires_fresh_scheduler_review" in result.stdout
    assert "planned_environment=ORCHJAX_GLOBAL_BATCH_ENTRY=benchmark_global_gpu_lifecycle.py" in result.stdout
    command_line = next(line for line in result.stdout.splitlines() if line.startswith("command="))
    tokens = shlex.split(command_line.split("=", 1)[1])
    coord_index = next(i for i, value in enumerate(tokens) if value.endswith("global_gpu_lifecycle_coordinator.py"))
    container_python = tokens.index("/apps/soft/sif/foundationpose") + 1
    assert tokens[container_python].endswith("/.venvs/orcjax_gpu/bin/python")
    tokens[container_python] = "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_gpu/bin/python"
    expanded = [sys.executable, tokens[coord_index], *tokens[coord_index + 1:]]
    env["ORCHJAX_GLOBAL_BATCH_ENTRY"] = "benchmark_global_gpu_lifecycle.py"
    env["SINGULARITYENV_ORCHJAX_GLOBAL_BATCH_ENTRY"] = "benchmark_global_gpu_lifecycle.py"
    coordinated = subprocess.run(expanded, cwd=ROOT, env=env, text=True, capture_output=True)
    assert coordinated.returncode == 0, coordinated.stderr
    report = json.loads(coordinated.stdout)
    assert report["status"] == "dry_run"
    assert report["strategies"] == ["phased_resident_control", "phased_release"]
    assert report["model_execution"] is False
    assert report["array_validation"] is False
    assert report["JAX_imported"] is False
    for strategy in report["strategies"]:
        output_dir = report["commands"][strategy][report["commands"][strategy].index("--output-dir") + 1]
        assert output_dir.replace("\\", "/").endswith(f"/{strategy}/capacity/B1024")
    assert report["commands"]["phased_resident_control"][
        report["commands"]["phased_resident_control"].index("--stage-shape") + 1
    ] == "1024"
    evidence_root = os.environ.get("ORCHJAX_LIFECYCLE_PREPARATION_EVIDENCE")
    if evidence_root:
        evidence_dir = Path(evidence_root)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        suffix = "spooled" if spooled else "checkout"
        (evidence_dir / f"launcher-dry-run-{suffix}.txt").write_text(result.stdout, encoding="utf-8")
        (evidence_dir / f"coordinator-dry-run-{suffix}.json").write_text(coordinated.stdout, encoding="utf-8")
    assert not (project / "runtime/outputs").exists()

    invocations = git_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 2
    for row in invocations:
        cwd, argv = row.split("\t", 1)
        assert cwd.replace("\\", "/").endswith("/runtime/worktrees/declared-code")
        assert not argv.startswith("-C ")
    assert invocations[0].endswith("rev-parse HEAD")
    assert invocations[1].split("\t", 1)[1] == "status --porcelain --untracked-files=all"
    launcher_source = (ROOT / "scripts/hpc/slurm_global_gpu_lifecycle.sh").read_text(encoding="utf-8")
    shared_log = "/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_gpu_lifecycle_%j.txt"
    assert f"#SBATCH -o {shared_log}" in launcher_source
    assert f"#SBATCH -e {shared_log}" in launcher_source
    assert "git -C" not in launcher_source
    assert "git status --porcelain --untracked-files=all" in launcher_source
    assert "--porcelain=v1" not in launcher_source


@pytest.mark.parametrize("changes, expected", [
    ({"EXPECTED_CODE_SHA": "bad"}, "EXPECTED_CODE_SHA"),
    ({"XLA_PYTHON_CLIENT_ALLOCATOR": "platform"}, "explicit XLA_PYTHON_CLIENT_ALLOCATOR overrides are forbidden"),
])
def test_launcher_rejects_invalid_code_pin_and_allocator(tmp_path, changes, expected):
    bash, _, _, launcher, _, env = fixture(tmp_path)
    env.update(changes)
    result = subprocess.run([bash, launcher.as_posix(), "--dry-run"], env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert expected in result.stderr
    assert "command=" not in result.stdout
