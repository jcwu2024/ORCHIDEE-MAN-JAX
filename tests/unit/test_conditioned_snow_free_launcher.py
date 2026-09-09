"""Execute both shell routes with fake cluster commands, then parse their actual CLI."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.hpc.run_conditioned_snow_free_continuation import _parser


@pytest.mark.parametrize('paid', [False, True])
def test_actual_wrapper_and_slurm_cli(tmp_path, paid):
    bash = shutil.which('bash')
    if bash is None:
        pytest.skip('requires Bash')
    root = Path(__file__).resolve().parents[2]
    hpc = tmp_path / 'scripts/hpc'
    hpc.mkdir(parents=True)
    names = ['run_conditioned_snow_free_continuation.sh', 'slurm_conditioned_snow_free_continuation.sh',
             'run_target_oriented_coupled_gpp_conditioned_attention_persistent_training.sh',
             'run_target_oriented_coupled_gpp_multi_group_direction.sh']
    for name in names:
        source = (root / 'scripts/hpc' / name).read_text()
        (hpc / name).write_text(source.replace('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX', tmp_path.as_posix()), newline='\n')
    output = tmp_path / 'runtime/outputs/check/report.json'
    output.parent.mkdir(parents=True)
    output.write_text('{}')  # Resume must be allowed in both launch routes.
    environment = dict(os.environ, WORKTREE=tmp_path.as_posix(), EXPECTED_GIT_HEAD='testhead',
                       OUTPUT=output.as_posix(), GPU_DEVICE='1', PREFLIGHT_ONLY='1',
                       FAKE_HOST='ibc13b03n01' if paid else 'gln01')
    if paid:
        environment.update(SLURM_JOB_ID='123', SLURM_JOB_PARTITION='gnall', CUDA_VISIBLE_DEVICES='3')
    else:
        for name in ('SLURM_JOB_ID', 'SLURM_JOB_PARTITION'):
            environment.pop(name, None)
    # Only external cluster commands are mocked; all shell branching/expansion is real.
    functions = '''
hostname() { printf '%s\n' "$FAKE_HOST"; }
git() { printf '%s\n' "$EXPECTED_GIT_HEAD"; }
test() { if [[ "$1" == -f || "$1" == -x ]]; then return 0; else builtin test "$@"; fi; }
readlink() { printf '/fake-driver\n'; }
nvidia-smi() { :; }
sha256sum() { :; }
env() { printf 'ARG=%s\n' "$@"; }
export -f hostname git test readlink nvidia-smi sha256sum env
bash "$1"
'''
    entry = hpc / ('slurm_conditioned_snow_free_continuation.sh' if paid else 'run_conditioned_snow_free_continuation.sh')
    result = subprocess.run([bash, '-c', functions, 'test-launch', entry.as_posix()],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    args = [line[4:] for line in result.stdout.splitlines() if line.startswith('ARG=')]
    module = args.index('-m')
    assert args[module + 1] == 'scripts.hpc.run_conditioned_snow_free_continuation'
    parsed = _parser().parse_args(args[module + 2:])
    assert parsed.preflight_only == (not paid)
    assert parsed.output == output
    assert f'SINGULARITYENV_CUDA_VISIBLE_DEVICES={3 if paid else 1}' in args
    if paid:
        assert '350m' in args
