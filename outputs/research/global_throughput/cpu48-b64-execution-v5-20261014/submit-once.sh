set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
E=$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v5-20261014
PREP=$P/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014
CODE=$P/runtime/worktrees/global-throughput-cpu48-b64-v5
cd "$P"
test -f "$E/presubmit-complete.txt"
test "$(sha256sum "$E/AUTHORIZATION.md" | cut -d' ' -f1)" = dd3c93acbb4e7d3e5c42d3eafd9d1a6055a0dfd817708a31aa9603cb79ac31b6
test "$(sha256sum "$PREP/PROPOSED_SUBMISSION.sh" | cut -d' ' -f1)" = c87ce9e9d274201c68980739748db33168fb8ae6b312ab390233a76bc968e993
test "$(sha256sum "$PREP/scalar-admission.json" | cut -d' ' -f1)" = 4d2d3d18d92c0d6c6bc737059bc2b2555cf36f77b4a3c4406844071ab6a3e7f5
test "$(git --git-dir="$CODE/.git" rev-parse HEAD)" = 0100c8b09be5137b822173e0a0be66cb64939727
test -z "$(cd "$CODE" && git status --porcelain --untracked-files=all)"
test -d "$P/runtime/logs" && test -w "$P/runtime/logs"
test ! -e "$E/submission-invoked.txt"
(set -o noclobber; date -Is > "$E/submission-invoked.txt")
printf 'submission_cwd=%s\nexecuted_proposal=%s\n' "$PWD" "$PREP/PROPOSED_SUBMISSION.sh" > "$E/submission-command.txt"
printf '%s\n' '["bash","/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014/PROPOSED_SUBMISSION.sh"]' > "$E/submission-argv.json"
set +e
bash "$PREP/PROPOSED_SUBMISSION.sh" > "$E/submission.stdout.raw" 2> "$E/submission.stderr.raw"
submission_code=$?
set -e
printf '%s\n' "$submission_code" > "$E/submission.exitcode.raw"
cat "$E/submission.stdout.raw"
cat "$E/submission.stderr.raw" >&2
printf 'submission_exit=%s\n' "$submission_code"
date -Is
exit "$submission_code"
