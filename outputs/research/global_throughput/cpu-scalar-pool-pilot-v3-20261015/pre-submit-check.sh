#!/bin/bash
set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
RT=$PROJECT/runtime
E=$RT/outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015
V=$RT/outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/v3
C=$RT/worktrees/global-throughput-cpu-scalar-pool-v3
S=/rmprog/slurm/v22.05.7/bin
[[ $(realpath -m "$E") == "$RT/outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015" ]]
mkdir -p "$E"
date -u +%FT%TZ
hostname
[[ ! -e $E/paid-submission-attempt.json ]]
cd "$C"
[[ $(git rev-parse HEAD) == bae03d9ef78726f00987737cdc58eac705a922ae ]]
[[ -z $(git status --porcelain --untracked-files=all) ]]
[[ $(git rev-parse HEAD:jax_orchidee) == 75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba ]]
printf '%s  %s\n' 259936bb09f5d0006895b1f5ae17fa298c0baada07edcb23b2355ed892ef796a "$V/scalar-admission.json" | sha256sum -c -
printf '%s  %s\n' db9a5a7bb4a382f373e7cf7cb4d340d185d32293a305a086c585dd6db57070f9 "$V/PROPOSED_SUBMISSION.sh" | sha256sum -c -
"$S/sacct" -j 15866807 --noheader --parsable2 --format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,NodeList | tee "$E/reused-free-verification.sacct.raw"
"$S/scontrol" show partition cnall | tee "$E/pre-submit-partition.txt"
"$S/sinfo" -p cnall -o '%P %a %l %D %t %C' | tee "$E/pre-submit-sinfo.txt"
"$S/squeue" -u liwei -o '%i %P %j %t %M %l %C %R' | tee "$E/pre-submit-squeue.txt"
"$PROJECT/.venvs/orcjax_cpu/bin/python" -B - "$V" <<'PY'
import json, sys
v = sys.argv[1]
r = json.load(open(v + '/target-verification.json'))
assert r['execution_commit'] == 'bae03d9ef78726f00987737cdc58eac705a922ae'
assert r['startup_matrix_passed'] and r['startup_test_count'] == 14
assert r['candidate_identity_matches']
assert r['reference_loading_evidence_reused_job'] == '15866448'
print('existing free startup/admission check matches; no science check repeated')
PY
bash "$V/PROPOSED_SUBMISSION.sh" --dry-run | tee "$E/approved-submission-dry-run.txt"
git rev-parse HEAD
printf 'pre-submit-check-passed\n'
