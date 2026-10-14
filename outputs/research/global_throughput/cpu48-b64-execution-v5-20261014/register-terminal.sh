set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
E=$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v5-20261014
/usr/bin/python - "$P" "$E" <<'PY'
from __future__ import print_function
import hashlib,json,os,sys
P,E=sys.argv[1:]
reg=P+'/runtime/deployment/registry.json'; md=P+'/runtime/deployment/README.md'
def read(path):
 with open(path,'rb') as f: return f.read()
def sha(data): return hashlib.sha256(data).hexdigest()
before,old_md=read(reg),read(md)
assert sha(before)=='7de57bf880156963def8a3faebadced52e5338f84d22c113d522f2d601406363'
assert sha(old_md)=='5e3ba3db3de0c7ea6fbd0136594b29afdf5f9df80aec072e2144ea16ebd45328'
key='latest_global_cpu48_b64_v5_execution'
old=json.loads(before); assert key not in old
entry=json.loads(read(E+'/terminal-registration-entry.json'))
assert entry['job_id']=='15857765' and entry['terminal_state']=='FAILED' and entry['parent_seconds']==819
assert sha(read(E+'/collector-result.json'))==entry['collector_result_sha256']
current=dict(old); current[key]=entry
new=(json.dumps(current,indent=2,ensure_ascii=True)+'\n').encode('ascii')
addition='''

CPU48 v5 terminal: sole approved paid job15857765 FAILED1:0/819s on
ibc12b03n02,48CPU/0GPU; exact0100c8 checkout remains clean. Actual48-core
admission, B64 compilation256.663s and first-call131.938s completed. Strict
CPU internal scalar gate failed three finite leaf comparisons at two points;
this is separate from the GPU3point/4comparison failures. Permutation/replacement
recorded gates passed, but reset/warm/validation seal were not reached. No valid
warm throughput. New whole-parent cost scenarios CNY0.546/0.7644 (unverified),
oldscalar15820467/2197s separate and not regenerated. Free singleCPU collector
15864638 completed0:0/6s and preserved failure accounting, without model rerun.
Retain raw candidate arrays and failed manifest under the existing v4 namespace
15857765; compact result/authorization/manifests are under v5 execution evidence.
This attempt and approval are closed; no repair, retry, extension or cancellation.
All old checkouts/refs/science assets and preparation bytes remain preserved.
'''
new_md=old_md+addition.encode('ascii')
for path,raw in ((reg+'.before-v5-terminal',before),(md+'.before-v5-terminal',old_md),
 (reg+'.v5-terminal.tmp',new),(md+'.v5-terminal.tmp',new_md)):
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 with os.fdopen(fd,'wb') as f: f.write(raw)
assert read(reg)==before and read(md)==old_md
os.rename(reg+'.v5-terminal.tmp',reg); os.rename(md+'.v5-terminal.tmp',md)
after=json.loads(read(reg)); assert dict((k,v) for k,v in after.items() if k!=key)==old
assert read(md).startswith(old_md)
receipt={'schema':'global_cpu48_v5_terminal_registry_v1','registry_before_sha256':sha(before),
 'registry_after_sha256':sha(read(reg)),'README_before_sha256':sha(old_md),
 'README_after_sha256':sha(read(md)),'old_records_and_bytes_preserved':True,
 'main_daily_research_refs_changed':False,'old_checkouts_or_assets_changed':False,
 'archive_or_delete':False,'entry':entry}
path=E+'/terminal-registry-receipt.json'; fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
with os.fdopen(fd,'wb') as f: f.write((json.dumps(receipt,indent=2)+'\n').encode('ascii'))
print(json.dumps(receipt,sort_keys=True))
PY
