set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
E=$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v5-20261014
R=$P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/15857765
S=/rmprog/slurm/v22.05.7/bin
STAMP=$(date +%Y%m%dT%H%M%S)
mkdir -p "$E/status"
date -Is > "$E/status/$STAMP.timestamp.txt"
"$S/squeue" -j 15857765 -o '%.18i %.9P %.22j %.2t %.10M %.6D %R' > "$E/status/$STAMP.squeue.txt"
"$S/scontrol" show job -dd 15857765 > "$E/status/$STAMP.scontrol.txt" 2> "$E/status/$STAMP.scontrol.stderr" || true
"$S/sacct" -j 15857765 --parsable2 --format=JobIDRaw,State,ExitCode,Elapsed,AllocCPUS,NodeList > "$E/status/$STAMP.sacct.txt"
cat "$E/status/$STAMP.squeue.txt" "$E/status/$STAMP.sacct.txt"
if [[ -f "$P/runtime/logs/global_cpu48_b64_v5_15857765.txt" ]]; then
  tail -c 10000 "$P/runtime/logs/global_cpu48_b64_v5_15857765.txt"
fi
if [[ -f "$R/batch.log" ]]; then tail -c 18000 "$R/batch.log"; fi
/usr/bin/python - "$R" <<'PY'
from __future__ import print_function
import json,os,sys
root=sys.argv[1]
for suffix in ('orchestration.json','batch/report.json'):
    path=root+'/'+suffix
    if not os.path.isfile(path): continue
    with open(path,'rb') as f: data=json.load(f)
    result={'path':path,'status':data.get('status'),'message':data.get('message'),
        'phases':data.get('phases',[])[-8:],'timing_keys':list(data.get('timing',{})),
        'gate_count':len(data.get('gates',[])), 'runtime_observation_sizes':
        dict((k,list(v)) for k,v in data.get('runtime_observations',{}).items())}
    print(json.dumps(result,sort_keys=True))
PY
