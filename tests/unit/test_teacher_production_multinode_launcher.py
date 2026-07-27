import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts" / "hpc" / "slurm_teacher_production_multinode.sh"
TASK = ROOT / "scripts" / "hpc" / "run_teacher_production_worker_task.sh"
FINALIZE = ROOT / "scripts" / "hpc" / "slurm_teacher_production_finalize.sh"


def test_multinode_shell_entry_points_are_executable_in_git():
    for path in (LAUNCHER, TASK):
        relative = path.relative_to(ROOT).as_posix()
        staged = subprocess.check_output(
            ["git", "ls-files", "--stage", relative],
            cwd=ROOT,
            text=True,
        )
        assert staged.startswith("100755 ")


def test_production_finalize_orders_all_fail_closed_gates():
    text = FINALIZE.read_text(encoding="utf-8")

    progress = text.index("teacher_shards progress")
    aggregate = text.index("teacher_shards aggregate")
    production = text.index("teacher_data_product_admission")
    acceptance = text.index("canonical_training_run")
    assert progress < aggregate < production < acceptance
    assert "--require-complete" in text
    assert "--require-production-dataset" in text


def test_multinode_launcher_has_safe_admission_defaults():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --partition=cnall" in text
    assert "#SBATCH --nodes=2" in text
    assert "#SBATCH --ntasks=10" in text
    assert "#SBATCH --ntasks-per-node=5" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --time=01:30:00" in text
    assert "--kill-on-bad-exit=0" in text
    assert "--wait=0" in text
    assert "teacher_${SLURM_JOB_ID}_worker_%t.txt" in text
    assert "SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required" in text
    assert 'git -C "$LAUNCHER_ROOT"' not in text
    assert 'cd "$LAUNCHER_ROOT" && git rev-parse HEAD' in text


def test_multinode_worker_maps_global_rank_and_staggers_by_local_rank():
    text = TASK.read_text(encoding="utf-8")

    assert 'WORKER_INDEX=${WORKER_INDICES[$SLURM_PROCID]}' in text
    assert "WORKER_INDEX=$((TEACHER_WORKER_OFFSET + SLURM_PROCID))" in text
    assert "STAGGER_SECONDS=$((SLURM_LOCALID * TEACHER_STARTUP_STAGGER_SECONDS))" in text
    assert "--worker-index \"$WORKER_INDEX\"" in text
    assert "--worker-count \"$TEACHER_WORKER_COUNT\"" in text
    assert "JAX_COMPILATION_CACHE_DIR=$CACHE_ROOT/worker-$WORKER_INDEX" in text


def test_multinode_launcher_supports_sparse_worker_recovery():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    task = TASK.read_text(encoding="utf-8")

    assert "TEACHER_WORKER_INDICES must be a comma-separated integer list" in launcher
    assert "TEACHER_WORKER_INDICES count must equal SLURM_NTASKS" in launcher
    assert "TEACHER_WORKER_INDICES contains a duplicate worker" in launcher
    assert 'IFS=\',\' read -r -a WORKER_INDICES <<< "$TEACHER_WORKER_INDICES"' in task


def test_multinode_worker_fails_closed_on_teacher_identity_and_paths():
    text = TASK.read_text(encoding="utf-8")

    assert "TEACHER_EXPECTED_GIT_HEAD" in text
    assert "Teacher worktree HEAD mismatch" in text
    assert 'test -z "$(git status --porcelain --untracked-files=all)"' in text
    assert '"$RUNTIME_ROOT"/*' in text
    assert '"$RUNTIME_ROOT/worktrees/"*' in text
