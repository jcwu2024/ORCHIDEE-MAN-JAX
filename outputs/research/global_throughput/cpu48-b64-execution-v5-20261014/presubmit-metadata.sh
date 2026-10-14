set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
E=$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v5-20261014
PREP=$P/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014
CODE=$P/runtime/worktrees/global-throughput-cpu48-b64-v5
S=/rmprog/slurm/v22.05.7/bin
test ! -e "$E"
mkdir "$E"
date -Is > "$E/presubmit-captured-at.txt"
hostname > "$E/presubmit-login-host.txt"
uptime > "$E/presubmit-login-uptime.txt"
"$S/sinfo" -N -p cnall -o '%N|%t|%C|%m|%G' > "$E/presubmit-sinfo.txt"
"$S/squeue" -p cnall -o '%.18i %.9P %.22j %.8u %.2t %.10M %.6D %R' > "$E/presubmit-queue.txt"
"$S/scontrol" show node > "$E/presubmit-node-facts.txt"
"$S/scontrol" show partition cnall > "$E/presubmit-partition.txt"
test -d "$P/runtime/logs" && test -w "$P/runtime/logs"
test -x "$P/.venvs/orcjax_cpu/bin/python"
test -f "$PREP/free-check/15857559/free-check-complete.txt"
cd "$CODE"
test "$(git rev-parse HEAD)" = 0100c8b09be5137b822173e0a0be66cb64939727
test "$(git rev-parse HEAD:jax_orchidee)" = 75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba
test -z "$(git status --porcelain --untracked-files=all)"
test ! -e .git/objects/info/alternates
test "$(git --git-dir="$P/runtime/deployment/repository.git" rev-parse refs/heads/codex/global-cpu48-b64-v5)" = 0100c8b09be5137b822173e0a0be66cb64939727
/usr/bin/python - "$P" "$E" "$PREP" "$CODE" <<'PY'
from __future__ import print_function
import hashlib,json,os,subprocess,sys
P,E,PREP,CODE=sys.argv[1:]
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(1048576)
            if not b: break
            h.update(b)
    return h.hexdigest()
expected={
 PREP+'/PROPOSED_SUBMISSION.sh':'c87ce9e9d274201c68980739748db33168fb8ae6b312ab390233a76bc968e993',
 PREP+'/execution-code-pin.json':'d800ffc9be9dc4867dc1b8ce19ca57b9da74082206f55b81d674a1adc556c2e6',
 PREP+'/scalar-admission.json':'4d2d3d18d92c0d6c6bc737059bc2b2555cf36f77b4a3c4406844071ab6a3e7f5',
 P+'/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl':'be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7',
 P+'/runtime/outputs/research/global_throughput/gpu-round4/model/stage_B4_group_references.pkl':'a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6',
 P+'/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/15820467/scalar/stage_B64_server_scalar_v1.pkl':'1cddeaf80c2fc36f173835a5ff1bd4d8d520b144d4c96289ef6fe86dde898dfa',
 CODE+'/scripts/dev/collect_global_cpu48_b64.py':'efcb32efa754bbdbb75936f358dc6cb5a286e6742948096bda3e3ba22d27a9e3'}
with open(PREP+'/execution-code-pin.json','rb') as f: pin=json.load(f)
expected.update((CODE+'/'+path,value) for path,value in pin['harness_blob_sha256'].items())
actual={}
for path,value in expected.items():
    actual[path]=digest(path)
    assert actual[path]==value, 'identity mismatch: '+path
for path in pin['changed_paths']:
    with open(CODE+'/'+path,'rb') as f: raw=f.read()
    assert raw==subprocess.check_output(['git','show','HEAD:'+path],cwd=CODE), 'blob mismatch: '+path
with open(PREP+'/scalar-admission.json','rb') as f: admission=json.load(f)
assert admission['candidate_identity']['execution_commit']==pin['execution_code_commit']
assert admission['candidate_identity']['execution_code_dir']==CODE
receipt={'schema':'global_cpu48_v5_presubmit_metadata_v1','execution_commit':pin['execution_code_commit'],
 'execution_code_dir':CODE,'code_clean':True,'no_alternates':True,'exact_hashes':actual,
 'log_directory_available':True,'canonical_python_executable_available':True,
 'runtime_evidence_reused':'free15857559 JAX/jaxlib0.4.38/x64; no JAX imported or model execution here',
 'scientific_source_tree_oid':pin['scientific_source']['science_tree_oid'],
 'placement':'Slurm automatic; no node/exclusive/memory override added','metadata_only':True,'paid_submitted':False}
with open(E+'/presubmit-identities.json','w') as f: json.dump(receipt,f,indent=2); f.write('\n')
print(json.dumps(receipt,sort_keys=True))
PY
date -Is > "$E/presubmit-complete.txt"
printf 'PRESUBMIT_METADATA=passed\n'
