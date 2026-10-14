#!/bin/bash
# Authorized preparation/deployment only: this script never calls sbatch/srun.
set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
RT=$PROJECT/runtime
DEPLOY=$RT/deployment
EVIDENCE=$RT/outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/v3
CODE=$RT/worktrees/global-throughput-cpu-scalar-pool-v3
TRANSFER=$RT/transfer/global-cpu-scalar-pool-v3-20261015
PYTHON=$PROJECT/.venvs/orcjax_cpu/bin/python
[[ $# == 1 ]]
for path in "$DEPLOY" "$EVIDENCE" "$CODE" "$TRANSFER"; do
  case "$(realpath -m "$path")" in "$RT"/*) ;; *) exit 2 ;; esac
done
[[ ! -e $CODE ]]
printf '%s  %s\n' "$1" "$EVIDENCE/execution-code-pin.json" | sha256sum -c -
read -r PIN BUNDLE_NAME BUNDLE_SHA ADMISSION_SHA PROPOSAL_SHA < <(
  "$PYTHON" -B - "$EVIDENCE/execution-code-pin.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p['paid_submitted'] is False and p['pilot']['authorized'] is False
assert p['base_execution_commit'] == '50864f065363f6cd0de685c18d9c257723bf2838'
print(p['execution_code_commit'], p['bundle']['name'], p['bundle']['sha256'],
      p['scalar_admission']['sha256'], p['proposed_submission']['sha256'])
PY
)
[[ $PIN =~ ^[0-9a-f]{40}$ ]]
BUNDLE=$TRANSFER/$BUNDLE_NAME
printf '%s  %s\n' "$BUNDLE_SHA" "$BUNDLE" | sha256sum -c -
printf '%s  %s\n' "$ADMISSION_SHA" "$EVIDENCE/scalar-admission.json" | sha256sum -c -
printf '%s  %s\n' "$PROPOSAL_SHA" "$EVIDENCE/PROPOSED_SUBMISSION.sh" | sha256sum -c -
git --git-dir="$DEPLOY/repository.git" bundle verify "$BUNDLE"
git --git-dir="$DEPLOY/repository.git" bundle list-heads "$BUNDLE"
git --git-dir="$DEPLOY/repository.git" fetch "$BUNDLE" refs/heads/codex/global-cpu-scalar-pool-v3:refs/heads/codex/global-cpu-scalar-pool-v3
[[ $(git --git-dir="$DEPLOY/repository.git" rev-parse refs/heads/codex/global-cpu-scalar-pool-v3) == "$PIN" ]]
git clone --no-hardlinks "$DEPLOY/repository.git" "$CODE"
cd "$CODE"
git checkout --detach "$PIN"
[[ $(git rev-parse HEAD) == "$PIN" ]]
[[ $(git rev-parse HEAD:jax_orchidee) == 75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba ]]
[[ -z $(git status --porcelain --untracked-files=all) ]]
[[ ! -s .git/objects/info/alternates ]]
"$PYTHON" -B - "$DEPLOY/registry.json" "$EVIDENCE" "$CODE" "$TRANSFER" <<'PY'
import datetime, hashlib, json, os, subprocess, sys
registry_path, evidence, code, transfer = sys.argv[1:]
before = open(registry_path, 'rb').read()
registry = json.loads(before)
key = 'global_cpu_scalar_pool_telemetry_fix_v3_20261015'
assert key not in registry, 'do not overwrite any previous deployment'
pin_bytes = open(os.path.join(evidence, 'execution-code-pin.json'), 'rb').read()
pin = json.loads(pin_bytes)
entry = {
    'schema': 'global_cpu_scalar_pool_telemetry_fix_deployment_v1',
    'utc': datetime.datetime.utcnow().isoformat() + 'Z',
    'execution_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code).decode().strip(),
    'execution_ref': pin['execution_ref'], 'code_dir': code,
    'science_tree_oid': pin['scientific_source']['science_tree_oid'],
    'checkout_state': 'detached_clean', 'alternate_dependencies': [],
    'transfer_dir': transfer, 'evidence_dir': evidence,
    'authorization': '2026-10-15 supervisor instruction: repair, deploy, free non-model verification, prepare exact pilot only',
    'proposed_allocation': dict(pin['pilot'], exclusive=False, memory_reservation=None, nodelist=None, no_requeue=True),
    'paid_submission_count': 0, 'paid_submissions_remaining': 0,
    'new_paid_approval_required': True,
    'pin_sha256': hashlib.sha256(pin_bytes).hexdigest(),
    'admission_sha256': pin['scalar_admission']['sha256'],
    'proposal_sha256': pin['proposed_submission']['sha256'],
    'registry_before_sha256': hashlib.sha256(before).hexdigest(),
    'preserve_all_prior_records_and_checkouts': True,
}
registry[key] = entry
open(os.path.join(evidence, 'registry-before.json'), 'wb').write(before)
open(os.path.join(evidence, 'deployment-receipt.json'), 'w').write(json.dumps(entry, indent=2, sort_keys=True) + '\n')
temporary = registry_path + '.scalar-pool-v3.tmp'
with open(temporary, 'x') as stream:
    json.dump(registry, stream, indent=2, sort_keys=True)
    stream.write('\n')
assert open(registry_path, 'rb').read() == before, 'concurrent registry edit; preserve temporary file'
os.rename(temporary, registry_path)
print(json.dumps(entry, sort_keys=True))
PY
date -u +%FT%TZ
git rev-parse HEAD
git status --short
