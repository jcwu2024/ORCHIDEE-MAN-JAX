set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
OUT=$P/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014
/usr/bin/python - "$P" "$OUT" <<'PY'
from __future__ import print_function
import hashlib,json,os,sys
P,OUT=sys.argv[1:]
registry=P+'/runtime/deployment/registry.json'
readme=P+'/runtime/deployment/README.md'
def read(path):
    with open(path,'rb') as f: return f.read()
def sha(raw): return hashlib.sha256(raw).hexdigest()
old_registry,old_readme=read(registry),read(readme)
assert sha(old_registry)=='a9ec87f475effcb5cc1a2afe9489566ce32bb310c27491565ed442fb15535b62'
assert sha(old_readme)=='4f8afd6c154c0d2af460dd7102dce43b2178e94bb2a999dd41e404a1cfd3c810'
old=json.loads(old_registry)
key='latest_global_cpu48_b64_v5_preparation'
assert key not in old
entry={'execution_commit':'0100c8b09be5137b822173e0a0be66cb64939727',
 'execution_ref':'codex/global-cpu48-b64-v5',
 'prerequisite_commit':'60dccbddde10be4295aab7e7fe55fa7864d3c3f3',
 'checkout':P+'/runtime/worktrees/global-throughput-cpu48-b64-v5',
 'evidence_dir':OUT,'checkout_clean':True,'git_alternates':False,
 'bundle':P+'/runtime/transfer/global-cpu48-b64-v5-0100c8b09be5/cpu48-b64-0100c8b09be5.bundle',
 'bundle_sha256':'0668675de188c67c23ebacaf5f52a722458badbc2d9c2117bf18bca288ba91c1',
 'pin_sha256':'d800ffc9be9dc4867dc1b8ce19ca57b9da74082206f55b81d674a1adc556c2e6',
 'admission_sha256':'4d2d3d18d92c0d6c6bc737059bc2b2555cf36f77b4a3c4406844071ab6a3e7f5',
 'proposal_sha256':'c87ce9e9d274201c68980739748db33168fb8ae6b312ab390233a76bc968e993',
 'free_job_id':'15857559','free_partition':'cntest','free_cpus':1,'free_time_limit_seconds':300,
 'free_state':'COMPLETED','free_exit_code':'0:0','free_elapsed_seconds':65,
 'free_report_sha256':'d70dd295df32227c6050db890c7732a1e54a6fe70bf7d5ea631a07b6a772247f',
 'reference_preparation':'passed','GPU_leaves':760,'CPU_scalar_leaves':760,
 'paid_submission':False,'paid_approval':False,'physical_lower_compile_run':False,
 'next_dispatch':'supervisor approval','retention':'Preserve v5 checkout/ref/bundle/evidence and all old checkouts, refs, scalar/GPU/prepared assets; no archival or deletion authorized.'}
assert sha(read(OUT+'/free-check/15857559/report.json'))==entry['free_report_sha256']
current=dict(old); current[key]=entry
new_registry=(json.dumps(current,indent=2,ensure_ascii=True)+'\n').encode('ascii')
addition='''

## CPU48 v5 prepared candidate, 2026-10-14

Retain `runtime/worktrees/global-throughput-cpu48-b64-v5` at
`0100c8b09be5137b822173e0a0be66cb64939727`, detached/clean with no alternates.
The six-path accepted identity repair leaves all 104 physical source blobs unchanged.
Retain ref `codex/global-cpu48-b64-v5`, its prerequisite v4 commit, transferred
bundle and `runtime/outputs/research/global_throughput/cpu48-b64-preparation-v5-20261014`.
Free job 15857559: cntest/ibtest02/1CPU/300s limit, COMPLETED 0:0 in65s;
real shared preparation loaded GPU760/scalar760 leaves, JAX/jaxlib0.4.38/x64,
matching candidate/stage identities. No physical runner/lower/compile/run occurred.
The target-shell proposal/launcher/coordinator/runner dry expansion passed.
Paid execution is not approved or submitted; next dispatch is supervisor approval.
Scalar61/64 GPU matches and existing scientific failures remain; no throughput claim.
Preserve all historical v1-v4 checkouts/refs, old scalar15820467, GPU/prepared data,
and closed-paid v4 evidence. No ref replacement, archive, cleanup or deletion here.
'''
new_readme=old_readme+addition.encode('ascii')
receipt_path=OUT+'/registry-retention-receipt.json'
assert not os.path.exists(receipt_path)
for path,raw in ((registry+'.before-v5-preparation',old_registry),(readme+'.before-v5-preparation',old_readme),
                 (registry+'.v5-preparation.tmp',new_registry),(readme+'.v5-preparation.tmp',new_readme)):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,'wb') as f: f.write(raw)
assert read(registry)==old_registry and read(readme)==old_readme
os.rename(registry+'.v5-preparation.tmp',registry)
os.rename(readme+'.v5-preparation.tmp',readme)
after=json.loads(read(registry)); assert after[key]==entry
assert dict((k,v) for k,v in after.items() if k!=key)==old
assert read(readme).startswith(old_readme)
receipt={'schema':'global_cpu48_v5_registry_retention_v1','registry_before_sha256':sha(old_registry),
 'registry_after_sha256':sha(read(registry)),'README_before_sha256':sha(old_readme),
 'README_after_sha256':sha(read(readme)),'old_records_preserved':True,
 'old_README_bytes_preserved':True,'new_key':key,'entry':entry,
 'main_daily_research_refs_changed':False,'old_checkout_or_asset_changed':False,
 'archive_or_delete':False,'paid_submission':False}
fd=os.open(receipt_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
with os.fdopen(fd,'wb') as f: f.write((json.dumps(receipt,indent=2)+'\n').encode('ascii'))
print(json.dumps(receipt,sort_keys=True))
PY
