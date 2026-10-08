import base64
import hashlib
import json
from pathlib import Path

folder = Path(__file__).resolve().parent
data = (folder / 'collect_b512_result.py').read_bytes()
helpers = {}
for name in ('gpu-signature-repair-20261008/collect_continuation_result.py', 'gpu-sweetspot-preparation-20261006/collect_capacity_result.py'):
    source = (folder.parent / name).read_bytes()
    helpers[name] = {'data': base64.b64encode(source).decode(), 'normalized_sha256': hashlib.sha256(source.replace(b'\r\n', b'\n')).hexdigest()}
runner = '''import base64,hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
started=time.perf_counter()
project=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
code=project/'runtime/worktrees/global-throughput-gpu-metadata-v1'
target=project/'runtime/outputs/research/global_throughput/gpu-metadata-repair-20261008/collect_b512_result.py'
data=base64.b64decode(PAYLOAD)
copied_dependencies=[]
for name,entry in json.loads(HELPERS).items():
    destination=project/'runtime/outputs/research/global_throughput'/name
    source=base64.b64decode(entry['data'])
    assert hashlib.sha256(source.replace(bytes([13,10]),bytes([10]))).hexdigest()==entry['normalized_sha256']
    if destination.exists():
        assert hashlib.sha256(destination.read_bytes().replace(bytes([13,10]),bytes([10]))).hexdigest()==entry['normalized_sha256'],'existing historical helper differs; never overwrite'
        action='existing historical helper verified; unchanged'
    else:
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(source)
        action='copied exact source bytes to confirmed-absent path'
    copied_dependencies.append({'path':str(destination),'action':action,'normalized_sha256':entry['normalized_sha256']})
assert hashlib.sha256(data).hexdigest()==EXPECTED_SHA
assert str(target.resolve()).startswith('/WORK/liwei_work/jcwu/')
target.parent.mkdir(parents=True,exist_ok=True)
if target.exists():
    assert target.read_bytes()==data,'refuse to overwrite a different external collector'
else:
    target.write_bytes(data)
namespace={'__name__':'collector_probe','__file__':str(target)}
exec(compile(data,str(target),'exec'),namespace)
sys.path.insert(0,str(code))
from scripts.dev.global_source_compatibility import source_identity
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()==namespace['CODE_SHA']
assert subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=code,text=True)==''
identity=source_identity(code)
assert identity==namespace['execution_identity']()
completed=subprocess.run(['/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python',str(target),'--help'],capture_output=True,text=True,timeout=20)
assert completed.returncode==0,completed.stderr
assert '--approved-wall-seconds' in completed.stdout and '--sacct-file' in completed.stdout
assert 'jax' not in sys.modules
assert subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=code,text=True)==''
print(json.dumps({'status':'passed','external_collector_path':str(target),'collector_raw_sha256':EXPECTED_SHA,
  'code_checkout':str(code),'code_commit':namespace['CODE_SHA'],'checkout_clean_before_after':True,
  'actual_execution_source_identity':identity,'legacy_dependency_paths':[str(namespace['prior'].__file__),str(namespace['prior'].legacy_path)],
  'legacy_dependency_hash_checks':'exact normalized source SHA checks passed during actual import',
  'dependency_copy_or_verification':copied_dependencies,
  'actual_cli_help_exit_code':completed.returncode,'help_stdout':completed.stdout,'help_stderr':completed.stderr,
  'JAX_imported':False,'model_preparation':False,'model_lowering':False,'model_compilation':False,'model_invocation':False,'GPU_initialization':False,
  'saved_origin_arrays_reread':False,'inner_wall_seconds':time.perf_counter()-started,
  'limitation':'stdlib source/CLI/dependency admission only; no new B512 result exists and terminal-job collection is not yet exercised'}),flush=True)
'''
prefix = 'PAYLOAD=' + repr(base64.b64encode(data).decode()) + '\nEXPECTED_SHA=' + repr(hashlib.sha256(data).hexdigest()) + '\nHELPERS=' + repr(json.dumps(helpers)) + '\n'
(folder / 'collector_interface_probe.py').write_text(prefix + runner)
