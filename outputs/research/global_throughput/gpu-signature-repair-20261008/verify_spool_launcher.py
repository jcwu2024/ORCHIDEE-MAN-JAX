"""Free launch-path diagnostic: dry-run only; never launches srun/model/GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

PROJECT = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
CODE = PROJECT / 'runtime/worktrees/global-throughput-gpu-sweetspot-v3'
EVIDENCE = PROJECT / 'runtime/outputs/research/global_throughput/gpu-signature-repair-20261008/spool-diagnostic-12b495b'
RUN = PROJECT / 'runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512'
assert not RUN.exists()
EVIDENCE.mkdir(parents=True, exist_ok=False)
spool = EVIDENCE / 'slurm-spool/job15643327'
spool.mkdir(parents=True)
copied = spool / 'slurm_script'
shutil.copyfile(CODE / 'scripts/hpc/slurm_global_gpu_capacity_continue.sh', copied)
assert not (spool / 'slurm_global_gpu_capacity.sh').exists()
env = os.environ.copy()
for key in ('CODE_DIR', 'SLURM_CPUS_PER_TASK', 'SLURM_JOB_GPUS', 'CUDA_VISIBLE_DEVICES',
            'CONTINUE_256_512', 'OPTIONAL_B64_BASELINE', 'B128_BASELINE_DIR'):
    env.pop(key, None)
env.update(RUN_ROOT=str(RUN), PYTHONDONTWRITEBYTECODE='1')
for mode in ('explicit_checkout', 'default_checkout'):
    if mode == 'explicit_checkout':
        env['CODE_DIR'] = str(CODE)
    else:
        env.pop('CODE_DIR', None)
    result = subprocess.run(['bash', str(copied), '--dry-run'], env=env,
                            text=True, capture_output=True, timeout=15, check=True)
    print(result.stdout, end='')
    tokens = shlex.split(next(row.removeprefix('command=') for row in result.stdout.splitlines()
                             if row.startswith('command=')))
    entry = next(i for i, token in enumerate(tokens) if token.endswith('global_batch_gpu_entry.py'))
    assert tokens[entry] == str(CODE / 'scripts/hpc/global_batch_gpu_entry.py')
    dry = subprocess.run([sys.executable, str(CODE / 'scripts/dev/benchmark_global_gpu_capacity.py'),
                          *tokens[entry + 1:], '--dry-run'], env=env,
                         text=True, capture_output=True, timeout=15, check=True)
    report = json.loads(dry.stdout)
    assert report['candidates'] == [256, 512]
    assert not report['B512_requires_speedup'] and not report['optional_B64']
    assert not report['JAX_imported'] and not report['model_execution']
    assert report['total_lanes'] == 512 and report['unique_real_points'] == 64
    print(json.dumps({'mode': mode, 'copied_wrapper_without_sibling': True,
                      'actual_Bash_to_Python_dry_run': report}))
bad = subprocess.run(['bash', str(copied), '--unexpected-argument'], env=env,
                     text=True, capture_output=True, timeout=15)
assert bad.returncode == 2 and 'usage:' in bad.stderr
assert not RUN.exists()
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip()
assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'],
                                   cwd=CODE, text=True).strip()
print(json.dumps({'status': 'passed', 'code_commit': commit, 'checkout_clean': True,
                  'argument_forwarding_rejection': 'passed', 'run_root_unused': True,
                  'model_execution': False, 'GPU_initialization': False}))
