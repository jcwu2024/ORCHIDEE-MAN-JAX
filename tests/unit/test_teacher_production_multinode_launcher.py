import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts" / "hpc" / "slurm_teacher_production_multinode.sh"
TASK = ROOT / "scripts" / "hpc" / "run_teacher_production_worker_task.sh"


def test_multinode_shell_entry_points_are_executable_in_git():
    for path in (LAUNCHER, TASK):
        relative = path.relative_to(ROOT).as_posix()
        staged = subprocess.check_output(
            ["git", "ls-files", "--stage", relative],
            cwd=ROOT,
            text=True,
        )
        assert staged.startswith("100755 ")


def test_multinode_launcher_has_safe_admission_defaults():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --partition=cnall" in text
    assert "#SBATCH --nodes=2" in text
    assert "#SBATCH --ntasks=10" in text
    assert "#SBATCH --ntasks-per-node=5" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --time=01:30:00" in text
    assert "--kill-on-bad-exit=0" in text
    assert "teacher_${SLURM_JOB_ID}_worker_%t.txt" in text


def test_multinode_worker_maps_global_rank_and_staggers_by_local_rank():
    text = TASK.read_text(encoding="utf-8")

    assert "WORKER_INDEX=$((TEACHER_WORKER_OFFSET + SLURM_PROCID))" in text
    assert "STAGGER_SECONDS=$((SLURM_LOCALID * TEACHER_STARTUP_STAGGER_SECONDS))" in text
    assert "--worker-index \"$WORKER_INDEX\"" in text
    assert "--worker-count \"$TEACHER_WORKER_COUNT\"" in text
    assert "JAX_COMPILATION_CACHE_DIR=$CACHE_ROOT/worker-$WORKER_INDEX" in text


def test_multinode_worker_fails_closed_on_teacher_identity_and_paths():
    text = TASK.read_text(encoding="utf-8")

    assert "TEACHER_EXPECTED_GIT_HEAD" in text
    assert "Teacher worktree HEAD mismatch" in text
    assert 'test -z "$(git status --porcelain --untracked-files=all)"' in text
    assert '"$RUNTIME_ROOT"/*' in text
    assert '"$RUNTIME_ROOT/worktrees/"*' in text
