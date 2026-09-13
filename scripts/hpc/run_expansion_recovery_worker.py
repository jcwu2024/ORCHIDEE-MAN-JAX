"""Resume assigned old workers using their existing committed data and manifests."""
import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


def run(args):
    plan = json.loads(args.plan.read_text())
    rank = int(os.environ.get('SLURM_PROCID', '0'))
    code = Path(os.environ['CODE_ROOT'])
    if plan['schema_version'] != 'one_day_expansion_recovery_v1':
        raise ValueError('unknown recovery plan')
    for item in plan['workers'][rank]['assignments']:
        env = dict(os.environ, START_ENTRY=str(item['first']), END_ENTRY=str(item['last']),
                   OUTPUT_DIR=str(Path(plan['original_root'])/'chunks'/f'worker-{item["old_worker"]}'),
                   RESUME='1', RUN_TIMEOUT='50m')
        command = ['bash', str(code/'scripts/hpc/run_expanded_one_day_worker.sh')]
        if args.dry_run:
            print(' '.join(f'{key}={shlex.quote(env[key])}' for key in
                           ('START_ENTRY', 'END_ENTRY', 'OUTPUT_DIR', 'RESUME')), flush=True)
            subprocess.run(command+['--dry-run'], env=env, check=True)
        else:
            print(f'recovery rank={rank} old_worker={item["old_worker"]} remaining={item["remaining"]}', flush=True)
            subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    run(parser.parse_args())
