"""Retrieve bounded free-check receipts; retain failed v2 and successful v3 separately."""
from pathlib import Path
import argparse
import subprocess
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
IO = BASE.parent / 'cpu-scalar-pool-pilot-20261014/cluster_io.py'
REMOTE = '/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/' + BASE.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=['v2', 'v3'], required=True)
    args = parser.parse_args()
    local = BASE if args.version == 'v2' else BASE / 'v3'
    remote = REMOTE if args.version == 'v2' else REMOTE + '/v3'
    names = ['free-allocation.txt', 'target-startup-matrix.json', 'free-verification.log',
             'free-verification-exit-code.txt', 'free-verification.started']
    if args.version == 'v3':
        names += ['target-verification.json', 'submission-target-dry-run.txt', 'launcher-target-dry-run.txt',
                  'parser-target-dry-run.stdout', 'parser-target-dry-run.stderr']
    for name in names:
        assert not (local / name).exists(), 'preserve any previously collected receipt'
        subprocess.run([sys.executable, '-B', '-X', 'utf8', str(IO), 'get', remote + '/' + name,
                        str(local / name)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
