set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
T=$P/runtime/transfer/global-cpu48-b64-v4-60dccbddde10
OUT=$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014
CODE=$P/runtime/worktrees/global-throughput-cpu48-b64-v4
cd "$P"
date -Is
hostname
test "$(sha256sum "$T/PROPOSED_SUBMISSION.sh" | cut -d' ' -f1)" = b86068c62c96464319175135e12e5d84ce2e494a00875aceaa01ee0b1a48c160
test "$(git --git-dir="$CODE/.git" rev-parse HEAD)" = 60dccbddde10be4295aab7e7fe55fa7864d3c3f3
test -z "$(cd "$CODE" && git status --porcelain --untracked-files=all)"
test -f "$OUT/preflight/15853600/preflight-complete.txt"
/rmprog/slurm/v22.05.7/bin/scontrol show node >"$OUT/presubmit-node-facts.txt"
/rmprog/slurm/v22.05.7/bin/squeue -p cnall -o '%.18i %.9P %.12j %.8u %.2t %.10M %.6D %R' >"$OUT/presubmit-queue.txt"
test ! -e "$OUT/submission-invoked.txt"
(set -o noclobber; date -Is >"$OUT/submission-invoked.txt")
printf 'executed_proposal=%s\nsubmission_cwd=%s\n' "$T/PROPOSED_SUBMISSION.sh" "$PWD"
set +e
bash "$T/PROPOSED_SUBMISSION.sh" >"$OUT/submission.stdout.raw" 2>"$OUT/submission.stderr.raw"
submission_code=$?
set -e
printf '%s\n' "$submission_code" >"$OUT/submission.exitcode.raw"
cat "$OUT/submission.stdout.raw"
cat "$OUT/submission.stderr.raw" >&2
printf 'submission_exit=%s\n' "$submission_code"
date -Is
exit "$submission_code"
