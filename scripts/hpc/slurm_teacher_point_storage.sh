#!/bin/bash
#SBATCH --job-name=orcjax-point-storage
#SBATCH --partition=cnmix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/teacher_point_storage_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/teacher_point_storage_%j.txt
#SBATCH --no-requeue

# Finite proposal only: override --time at sbatch after explicit cost approval.
# 1 allocated CPU * 12h * CNY0.07/core-hour = CNY0.84 worst case.
# Required: --checkout ABSOLUTE_CHECKOUT and all required batch CLI arguments.
# --dry-run prints the exact expanded command and performs no filesystem writes.
set -euo pipefail

CHECKOUT=''
DRY_RUN=0
BATCH_ARGS=()
while (($#)); do
    case "$1" in
        --checkout)
            (($# >= 2)) || { echo 'missing --checkout value' >&2; exit 2; }
            CHECKOUT=$2; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) BATCH_ARGS+=("$1"); shift ;;
    esac
done
case "$CHECKOUT" in
    /WORK/liwei_work/jcwu/*) ;;
    *) echo '--checkout must name the exact checkout under /WORK/liwei_work/jcwu/' >&2; exit 2 ;;
esac
PYTHON=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python
COMMAND=(nice -n 15 "$PYTHON" -B -m research.daily_coarse_graining.teacher_point_batch "${BATCH_ARGS[@]}")
if ((DRY_RUN)); then
    printf 'cwd=%q\n' "$CHECKOUT"
    printf '%q ' "${COMMAND[@]}" --dry-run
    printf '\n'
    exit 0
fi

[[ ${SLURM_JOB_PARTITION:-} == cnmix && -n ${SLURM_JOB_ID:-} ]] || {
    echo 'submit only to an explicitly approved cnmix allocation' >&2; exit 2;
}
test -x "$PYTHON"
cd "$CHECKOUT"
[[ $(pwd -P) == "$CHECKOUT" ]] || { echo 'checkout resolved to a different directory' >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
printf 'host=%s job=%s partition=%s cpus=%s load=%s\n' "$(hostname)" "$SLURM_JOB_ID" "$SLURM_JOB_PARTITION" "${SLURM_CPUS_PER_TASK:-}" "$(cat /proc/loadavg)"
grep -E '^(Cpus_allowed_list|Mems_allowed_list):' /proc/self/status
exec "${COMMAND[@]}"
