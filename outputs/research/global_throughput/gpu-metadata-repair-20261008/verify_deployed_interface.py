"""Bounded stdlib-only verification of the immutable B512 proposal checkout."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
ROOT = PROJECT / 'runtime/worktrees/global-throughput-gpu-metadata-v1'
EXPECTED = 'ad6c0e1faa7e22b590627844e9cb0ad492e9deb3'
EVIDENCE = PROJECT / 'runtime/outputs/research/global_throughput/gpu-metadata-repair-20261008/deployment-ad6c0e1'
assert EVIDENCE.resolve().is_relative_to(PROJECT.resolve())
assert not EVIDENCE.exists(), 'retain existing receipt; do not overwrite'
EVIDENCE.mkdir(parents=True)
start = time.perf_counter()
allowed = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0, set(allowed[:4]))
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                 MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CODE_DIR=str(ROOT))
receipt = {'status': 'running', 'code_checkout': str(ROOT), 'expected_commit': EXPECTED,
           'CPU_affinity': sorted(os.sched_getaffinity(0)), 'commands': [],
           'model_preparation': False, 'model_lower_compile_execution': False,
           'GPU_initialization': False, 'new_paid_jobs': 0}


def run(name, command, cwd=ROOT):
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False,
                            stdin=subprocess.DEVNULL, timeout=20)
    receipt['commands'].append({'name': name, 'command': [str(v) for v in command],
                                'exit_code': result.returncode,
                                'stdout': result.stdout, 'stderr': result.stderr})
    assert result.returncode == 0, (name, result.stderr)
    return result.stdout


try:
    actual_commit = run('code_commit', ['git', 'rev-parse', 'HEAD']).strip()
    assert actual_commit == EXPECTED
    assert not run('clean_before', ['git', 'status', '--porcelain', '--untracked-files=all']).strip()
    receipt['actual_commit'] = actual_commit
    receipt['python'] = sys.version
    receipt['jax_distribution'] = importlib.metadata.version('jax')
    assert receipt['jax_distribution'] == '0.4.38'
    sys.path.insert(0, str(ROOT))
    from scripts.dev.global_source_compatibility import source_identity

    source = source_identity(ROOT)
    assert source['execution_source_compatibility']['active']
    assert source['science_sha256'] == '11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5'
    assert source['execution_source_compatibility']['baseline_checkout_bytes_sha256'] == source['science_sha256']
    assert source['actual_science_source_sha256'] != source['science_sha256']
    receipt['source_identity'] = source
    prepared = PROJECT / 'runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl'
    full = PROJECT / 'runtime/outputs/research/global_throughput/gpu-round4/model'
    memory = PROJECT / 'runtime/outputs/research/global_throughput/server-benchmark/global-throughput-v2-2b14506/gpu-b64/candidate-first'
    b128 = PROJECT / 'runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v1-load512/capacity'
    b256 = PROJECT / 'runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512/capacity'
    config = ROOT / 'configs/orchidee_man_250919.yaml'
    assets = PROJECT / 'runtime/outputs/research/global_throughput/round4-inputs/selected64-assets-v2'
    entry = ROOT / 'scripts/dev/benchmark_global_gpu_capacity.py'
    cuda_entry = ROOT / 'scripts/hpc/global_batch_gpu_entry.py'
    for path in (prepared, full, memory, b128, b256, config, assets, entry, cuda_entry,
                 PROJECT / '.venvs/orcjax_gpu/bin/python', Path('/apps/soft/sif/foundationpose')):
        assert path.exists(), str(path)
    receipt['declared_source_paths_exist'] = True
    run('help', [sys.executable, str(entry), '--help'])
    output = run('capacity_actual_cli', [sys.executable, str(entry), '--dry-run', '--continue-512',
        '--prepared-input', str(prepared), '--full-reference-dir', str(full), '--memory-baseline-dir', str(memory),
        '--b128-baseline-dir', str(b128), '--b256-baseline-dir', str(b256),
        '--output-dir', str(EVIDENCE / 'prospective-capacity'), '--config', str(config),
        '--asset-root', str(assets), '--warm-repeats', '5', '--budget-seconds', '2700'])
    parsed = json.loads(output)
    assert parsed['candidates'] == [512] and parsed['budget_seconds'] == 2700
    assert parsed['JAX_imported'] is False and parsed['model_execution'] is False
    receipt['capacity_dry_run'] = parsed
    wrapper = ROOT / 'scripts/hpc/slurm_global_gpu_capacity_512.sh'
    shared = ROOT / 'scripts/hpc/slurm_global_gpu_capacity.sh'
    run('bash_syntax_wrapper', ['bash', '-n', str(wrapper)])
    run('bash_syntax_shared', ['bash', '-n', str(shared)])
    direct = run('declared_checkout_wrapper', ['bash', str(wrapper), '--dry-run'])
    spool = EVIDENCE / 'slurm-spool-wrapper.sh'
    shutil.copyfile(wrapper, spool)
    copied = run('copied_Slurm_spool_wrapper', ['bash', str(spool), '--dry-run'])
    assert direct == copied
    assert '--continue-512' in direct and '--continue-256-512' not in direct
    assert '--budget-seconds 2700' in direct and '--warm-repeats 5' in direct
    assert str(cuda_entry) in direct and 'wall=00:45:00' in direct
    assert not run('clean_after', ['git', 'status', '--porcelain', '--untracked-files=all']).strip()
    assert 'jax' not in sys.modules
    receipt.update(status='passed', JAX_imported=False, direct_spool_expansion_equal=True,
                   unchanged_CUDA_entry_runtime='pending GPU job; path exists but not invoked')
except Exception as error:
    receipt.update(status='failed', error=repr(error))
    raise
finally:
    receipt['outer_wall_seconds'] = time.perf_counter() - start
    (EVIDENCE / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
