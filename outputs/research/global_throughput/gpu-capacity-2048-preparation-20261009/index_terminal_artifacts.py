import hashlib,json,time
from pathlib import Path
project=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
run=project/'runtime/outputs/research/global_throughput/server-benchmark/gpu-fixed2048-v1'
prep=project/'runtime/outputs/research/global_throughput/gpu-capacity-2048-preparation-20261009'
collected=json.loads((prep/'collected-fixed2048.json').read_text())
control=collected['valid_exploratory_stages'][0]
reused={'capacity/B1024/stage_capacity_B1024_fixed2048_fresh_v1.pkl':control['checkpoint_sha256'],
 'capacity/B1024/report.json':control['report_sha256'],'capacity/B1024/stage_manifest.json':control['manifest_sha256']}
files={}
for f in sorted(run.rglob('*')):
 if not f.is_file():continue
 rel=f.relative_to(run).as_posix()
 if rel in reused:
  sha=reused[rel];method='completed_strict_collector_full_SHA256_reused_no_second_large_hash'
 else:
  sha=hashlib.sha256(f.read_bytes()).hexdigest();method='test02_stdlib_SHA256_small_terminal_metadata'
 files['job-15671732-saved-result/'+rel]={'remote_path':str(f),'bytes':f.stat().st_size,'sha256':sha,'hash_method':method}
for name,source in [('job-15671732-terminal-log.txt',project/'runtime/logs/global_gpu_capacity_2048_15671732.txt'),('job-15671732-sacct-raw9.txt',prep/'job-15671732-sacct-raw9.txt'),('collected-fixed2048.json',prep/'collected-fixed2048.json'),('job-15671732-collector-log.txt',prep/'job-15671732-collector-log.txt')]:
 files[name]={'remote_path':str(source),'bytes':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'hash_method':'test02_stdlib_SHA256_small_terminal_metadata'}
result={'job_id':'15671732','scope':'complete terminal raw artifacts; large checkpoint SHA/manifest/report proof reused from successful strict saved-array collector; no model/GPU call','files':files,'B2048_checkpoint_exists':False,'hashing_host':'test02','affinity':[0,1,2,3]}
(prep/'terminal-artifact-index.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'files':len(files),'checkpoint_bytes':control['checkpoint_bytes'],'large_checkpoint_rehashed':False,'B2048_checkpoint_exists':False}))
