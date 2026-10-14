set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
E=$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v5-20261014
R=$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/15857765
/rmprog/slurm/v22.05.7/bin/sacct -j 15864638 --parsable2 --units=K --format=JobIDRaw,Partition,State,ExitCode,Elapsed,Timelimit,AllocCPUS,NodeList > "$E/terminal/collector.sacct.raw"
/usr/bin/python - "$E" "$R" <<'PY'
from __future__ import print_function
import hashlib,json,os,sys
E,R=sys.argv[1:]
def file_record(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  while True:
   b=f.read(1048576)
   if not b: break
   h.update(b)
 return {'path':path,'bytes':os.path.getsize(path),'sha256':h.hexdigest(),'retained_remote':True}
files=[]
for root,dirs,names in os.walk(R):
 for name in sorted(names): files.append(file_record(os.path.join(root,name)))
receipt={'schema':'global_cpu48_v5_saved_runtime_artifacts_v1','run_root':R,'files':files,
 'arrays_inspected':False,'model_recomputation':False}
with open(E+'/saved-runtime-artifacts.json','w') as f: json.dump(receipt,f,indent=2); f.write('\n')
print(json.dumps(receipt,sort_keys=True))
PY
