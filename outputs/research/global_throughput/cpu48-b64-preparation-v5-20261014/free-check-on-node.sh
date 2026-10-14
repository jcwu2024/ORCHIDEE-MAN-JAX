set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
OUT=$P/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014
CODE=$P/runtime/worktrees/global-throughput-cpu48-b64-v5
T=$P/runtime/transfer/global-cpu48-b64-v5-0100c8b09be5
RUN=$OUT/free-check/$SLURM_JOB_ID
test ! -e "$RUN"
mkdir -p "$RUN"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=true ORCHJAX_DISABLE_COMPILATION_CACHE=1
unset XLA_FLAGS
cd "$CODE"
/rmprog/slurm/v22.05.7/bin/scontrol show job -dd "$SLURM_JOB_ID" > "$RUN/actual-free-allocation.txt"
bash -n "$OUT/PROPOSED_SUBMISSION.sh"
bash -n "$CODE/scripts/hpc/slurm_global_server_cpu_48core.sh"
test "$SLURM_CPUS_PER_TASK" = 1
env -u SLURM_CPUS_PER_TASK bash "$OUT/PROPOSED_SUBMISSION.sh" --dry-run > "$RUN/launcher-dry-run.stdout" 2> "$RUN/launcher-dry-run.stderr"
PY=$P/.venvs/orcjax_cpu/bin/python
"$PY" -B "$T/check-target-metadata.py" "$OUT" "$CODE" > "$RUN/package-and-arguments.stdout" 2> "$RUN/package-and-arguments.stderr"
ARGS=("$PY" -B "$CODE/scripts/dev/diagnose_global_cpu_references.py"
 --mode batch --cpu-budget 1 --sizes 64
 --prepared-input "$P/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl"
 --gpu-reference-dir "$P/runtime/outputs/research/global_throughput/gpu-round4/model"
 --expected-gpu-reference-sha256 a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6
 --cpu-reference-dir "$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/15820467/scalar"
 --cpu-reference-admission-record "$OUT/scalar-admission.json"
 --expected-cpu-reference-admission-sha256 4d2d3d18d92c0d6c6bc737059bc2b2555cf36f77b4a3c4406844071ab6a3e7f5
 --validation-mode exploratory --cpu-reference-role cpu-internal --warm-repeats 5
 --memory-guard-gib 64 --config "$CODE/configs/orchidee_man_250919.yaml" --truth-root "$CODE" --output-dir "$RUN")
printf '%s\n' "${ARGS[@]}" > "$RUN/actual-argv.txt"
set +e
"${ARGS[@]}" > "$RUN/stdout.raw" 2> "$RUN/stderr.raw"
EXIT_CODE=$?
set -e
printf '%s\n' "$EXIT_CODE" > "$RUN/exitcode.raw"
cat "$RUN/stdout.raw"
cat "$RUN/stderr.raw" >&2
sha256sum "$RUN/report.json" "$OUT/scalar-admission.json" "$OUT/PROPOSED_SUBMISSION.sh" "$OUT/execution-code-pin.json"
git rev-parse HEAD HEAD:jax_orchidee
test -z "$(git status --porcelain --untracked-files=all)"
test "$EXIT_CODE" = 0
date -Is > "$RUN/free-check-complete.txt"
