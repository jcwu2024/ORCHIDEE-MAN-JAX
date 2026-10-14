set -euo pipefail
PHASE=$1
test "$PHASE" = before || test "$PHASE" = after
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE=$P/runtime/diagnostics/global-cpu-reference-identity-20261014
OUT=$P/runtime/outputs/research/global_throughput/cpu-reference-identity-repair-20261014
RUN=$OUT/$PHASE
test ! -e "$RUN"
mkdir "$RUN"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=true ORCHJAX_DISABLE_COMPILATION_CACHE=1
unset XLA_FLAGS
cd "$CODE"
git rev-parse HEAD HEAD:jax_orchidee > "$RUN/source-identity.txt"
git status --porcelain --untracked-files=all > "$RUN/overlays.txt"
sha256sum scripts/dev/benchmark_global_batch.py scripts/dev/benchmark_global_server_cpu.py scripts/dev/diagnose_global_cpu_references.py > "$RUN/overlay-hashes.txt"
sha256sum "$OUT/original-admission.json" "$OUT/diagnostic-admission.json" > "$RUN/admission-hashes.txt"
test "$(sha256sum "$OUT/diagnostic-admission.json" | cut -d' ' -f1)" = 41f59ab72cd620cd892398ae3b069d1867b6b7d72cd98dede40dbf93c0cdde51
/rmprog/slurm/v22.05.7/bin/scontrol show job -dd "$SLURM_JOB_ID" > "$RUN/actual-free-allocation.txt"
PY=$P/.venvs/orcjax_cpu/bin/python
ARGS=("$PY" -B "$CODE/scripts/dev/diagnose_global_cpu_references.py"
 --mode batch --cpu-budget 1 --sizes 64
 --prepared-input "$P/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl"
 --gpu-reference-dir "$P/runtime/outputs/research/global_throughput/gpu-round4/model"
 --expected-gpu-reference-sha256 a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6
 --cpu-reference-dir "$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/15820467/scalar"
 --cpu-reference-admission-record "$OUT/diagnostic-admission.json"
 --expected-cpu-reference-admission-sha256 41f59ab72cd620cd892398ae3b069d1867b6b7d72cd98dede40dbf93c0cdde51
 --validation-mode exploratory --cpu-reference-role cpu-internal --warm-repeats 5
 --memory-guard-gib 64 --config "$CODE/configs/orchidee_man_250919.yaml" --truth-root "$CODE"
 --output-dir "$RUN")
printf '%s\n' "${ARGS[@]}" > "$RUN/actual-argv.txt"
set +e
"${ARGS[@]}" > "$RUN/stdout.raw" 2> "$RUN/stderr.raw"
CODE_EXIT=$?
set -e
printf '%s\n' "$CODE_EXIT" > "$RUN/exitcode.raw"
cat "$RUN/stdout.raw"
cat "$RUN/stderr.raw" >&2
test -z "$(git diff --name-only -- jax_orchidee)"
cd "$P/runtime/worktrees/global-throughput-cpu48-b64-v4"
test -z "$(git status --porcelain --untracked-files=all)"
if test "$PHASE" = before; then test "$CODE_EXIT" = 1; else test "$CODE_EXIT" = 0; fi
