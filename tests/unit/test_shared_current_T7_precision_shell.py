"""Actual shell argument expansion only; no Slurm or runtime child execution."""
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SLURM = ROOT/'scripts/hpc/slurm_shared_current_T7_precision_admission.sh'
WRAPPER = ROOT/'scripts/hpc/run_shared_current_T7_precision_admission.sh'


def bash():
    git_bash = Path('C:/Program Files/Git/bin/bash.exe')
    value = str(git_bash) if git_bash.exists() else shutil.which('bash')
    if value is None:
        pytest.skip('Bash unavailable; actual target expansion remains required')
    return value


def test_actual_slurm_dryrun_source_cache_and_wrapper_forwarding():
    result = subprocess.run([bash(), str(SLURM), '--dry-run'], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    assert len(lines) == 2 and shlex.split(lines[0])[0] == 'sbatch'
    command = shlex.split(lines[1])
    assert command[:2] == ['bash', '/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-current-model-precision-admission-20261014-v2/scripts/hpc/run_shared_current_T7_precision_admission.sh']
    assert command.count('--source-cache') == 1
    source_cache = command[command.index('--source-cache')+1]
    assert source_cache == '/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-current-model-precision-admission-20261014-v2/outputs/T7-current-model-precision-admission-20261014-v2/source-cache'
    # Wrapper --dry-run prints expanded outer CPU argv; it never launches it.
    wrapper = subprocess.run([bash(), str(WRAPPER), '--dry-run', *command[2:]], cwd=ROOT,
                             capture_output=True, text=True, check=True)
    argv = shlex.split(wrapper.stdout.strip())
    assert argv[1] == '-B' and argv[2].endswith('/scripts/hpc/run_shared_current_T7_precision_admission.py')
    assert argv[3:] == command[2:]
    from scripts.hpc.probe_shared_current_T7_precision_admission import parser
    with pytest.raises(SystemExit) as error:
        parser().parse_args(argv[3:])
    assert error.value.code == 2


def test_wrapper_preserves_explicit_source_cache_with_spaces():
    value = '/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/task/outputs/task cache/source-cache'
    result = subprocess.run([bash(), str(WRAPPER), '--dry-run', '--source-cache', value], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    argv = shlex.split(result.stdout.strip())
    assert argv[3:] == ['--source-cache', value]


@pytest.mark.parametrize('value,expected', [('1792288108.2933187', 0), ('', 2), ('nan', 2), ('1;exit 0', 2)])
def test_container_deadline_export_rejects_missing_or_malformed_value(value, expected):
    import os
    source = WRAPPER.read_text()
    start = source.index('  if [[ $ENTRY == "$CODE/scripts/hpc/probe_shared_current_T7_precision_admission.py" ]]; then', source.index('export SINGULARITYENV_CURRENT_T7_SHELL_STARTED_EPOCH'))
    end = source.index('  fi', start) + len('  fi')
    code = source[start:end] + '\n[[ $SINGULARITYENV_CURRENT_T7_PROBE_DEADLINE_EPOCH == "$CURRENT_T7_PROBE_DEADLINE_EPOCH" ]]'
    env = dict(os.environ, CODE='/code', ENTRY='/code/scripts/hpc/probe_shared_current_T7_precision_admission.py', CURRENT_T7_PROBE_DEADLINE_EPOCH=value)
    result = subprocess.run([bash(), '-c', code], env=env, capture_output=True, text=True)
    assert result.returncode == expected
